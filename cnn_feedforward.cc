#include <stdio.h>
#include <string.h>
#include <time.h>
#include <string>
#include <sstream>

#include <pthread.h>
#include <unistd.h>

#include <mkl.h>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime_api.h>

#include "gpuErrCheck_INLINE_H_.h"
#include "lbx_cuda_kernels.h"
#include "cnn_io.h"
#include "lbx_nn.h"
#include "parse-options.h"
#include "cnnFunc.h"

pthread_mutex_t train_data_mutex;
pthread_mutex_t cpu_model_mutex;
pthread_mutex_t xent_mutex;

FILE 	*file_pointer_key;
FILE 	*file_pointer_label;
FILE 	*file_pointer_value;
FILE    *file_out_pointer;
long long int total_size;

int 	size_row_left_to_read_this_utter;
bool 	finished_reading_a_utter;

typedef struct {

    int gpu_id_s;

    bool cv_flag_s;

    float *nn_para_cpu_s;

    int nn_layer_num_s;
    int nn_para_num_s;

    float *cnn_cpu_s;
    cnn_params *params_cnn_s;
    int cnn_hidden_node_num_s;

    int *nn_in_dim_s;
    int *nn_out_dim_s;

    int frame_length_s;
    int splice_length_s;
    int *splice_index_s; 

    int nn_in_stride_s;
    int nn_out_stride_s;

    float learn_rate_s;
    int batch_size_s;
    int updates;
    int train_data_buffer_size_s;

    float *add_row_data_cpu_s;
    float *mul_row_data_cpu_s;

    long long *utter_all_merge_s;
    long long *frames_all_merge_s;

    float *loss_all_merge_s;
    long long *correct_all_merge_s;

    bool multicard;

} lbx_pthread_arg;

void *lbxParallelTrainLock ( void * );


// main parameter requirement:
// 1 bool		cv_flag
// 2 char		file_pointer_key (dir)
// 3 char		file_out_pointer (dir)
// 4 char		file_pointer_featrure_transform (dir)
// 5 float		learn_rate
// 6 int 		batch_size
// 7 int 		train_data_buffer_size
// 8 char		file_pointer_nn_para (dir)
// 9 char		file_pointer_nn_para_out (dir)

int main(int argc, char const *argv[]) {
    const char* usage = 
	"perform fast iterations of Neural Network training by SGD.\n";
    int bunchsize = 512;
    int cachesize = 32768;
    int num_threads = -1;
    int num_updates = 3;
    float learn_rate = 0.008;
    bool crossvalidate = false;
    std::string feature_transform;
    std::string model_filename;
    std::string features;
    std::string alignments;
    std::string target_model_filename;
    ParseOptions po(usage);
    po.Register("bunchsize", &bunchsize, "Size of weight update block");
    po.Register("cachesize", &cachesize, "Size of cache for frame level shuffling");
    po.Register("learn-rate", &learn_rate, "Learning rate");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("feature-transform", &feature_transform, "Feature transform Neural Network");
    po.Register("num-threads", &num_threads, "Number of threads to be used.");
    po.Register("num-updates", &num_updates, "frequency of updates per threads.");
    po.Read( argc,argv );
    if( po.NumArgs() != 5 - (crossvalidate?1:0)){
	po.PrintUsage();
	exit(1);
    }

    model_filename = po.GetArg(1);
    features = po.GetArg(2);
    alignments = po.GetArg(3);
    std::string size_str = po.GetArg(4);
    std::stringstream sstr(size_str);
    sstr >> total_size;

    if( !crossvalidate )
	target_model_filename = po.GetArg(4);
    //	if( argc < 9 ) {
    //		printf("Illegal Input! Input Num = %d\n", argc);
    //		for( int i=1; i<argc; i++ ) printf("%s\n", argv[i]);
    //		exit(-1);
    //	}
    //	
    //	bool cv_flag = atoi( argv[1] );
    //	if( !cv_flag && argc < 10 ) {
    //		printf("Illegal Input for Training! Input Num = %d\n", argc);
    //		for( int i=1; i<argc; i++ ) printf("%s\n", argv[i]);
    //		exit(-1);
    //	}

    printf( "-------------- LBX LOG --------------\n" );

    if(crossvalidate) 
	printf("           Cross Validation\n");
    else 
	printf("             NN Training\n");

    printf( "-------------------------------------\n" );

    time_t start_time, finish_time;
    long long total_time;

    printf( "Feats: %s\n", features.c_str() );
    file_pointer_key = fopen(features.c_str(), "rb");

    if( file_pointer_key == NULL ) {
	printf("KEY FILE OPEN ERROR!\n");
	return -2;
    }

    printf( "Output File: %s\n", alignments.c_str() );
    file_out_pointer = fopen(alignments.c_str(), "wb");

    if( file_out_pointer== NULL ) {
	printf("LABEL FILE OPEN ERROR!\n");
	return -2;
    }

    float *add_row_data, *mul_row_data;

    int feature_in_dim, feature_out_dim;

    int splice_length;
    int *splice_index;

    printf( "Feature_Transform: %s\n", feature_transform.c_str() );

    if( lbxReadFeatureTransform( feature_transform.c_str(),
		&feature_in_dim, &feature_out_dim,
		&splice_length,
		&splice_index,
		&add_row_data, 
		&mul_row_data ) )

	return -2;

    printf( "-------------------------------------\n" );

    // ------------------ //
    // Read NN Parameters
    // ------------------ //

    int train_data_buffer_size = cachesize;

    printf( "Learn Rate = %f\n", learn_rate );
    printf( "Batch Size = %d\n", bunchsize );
    printf( "Buffer Size = %d\n", cachesize );
    printf( "Updates Freq = %d\n", num_updates );
    printf( "-------------------------------------\n" );

    printf( "Load NN Parameter FILE: %s\n", model_filename.c_str() );

    float *nn_para_cpu;
    int *nn_in_dim, *nn_out_dim;
    int nn_layer_num;

    float *cnn_cpu;
    cnn_params params_cnn;
    int cnn_hidden_node_num;


    nn_layer_num = ReadKaldiCNNnet( model_filename.c_str(), 
	    &nn_para_cpu, &cnn_cpu,
	    &nn_in_dim, &nn_out_dim, &params_cnn );

    if( nn_layer_num <= 0 ) {
	printf("Read Nnet File Error!\n");
	return - 2;
    }

    int layer_count;

    int nn_para_num = 0;
    for (layer_count = 0; layer_count < nn_layer_num; layer_count++) {
	nn_para_num += ( nn_in_dim[layer_count] +1 )* nn_out_dim[layer_count];
    }
    // std::cout<< "Hidden Neuron Num = " << nn_hidden_node_num << std::endl;

    cnn_hidden_node_num = params_cnn.cnn_out_dim[0];

    printf("CNN files read\n");

    ///******************************************************************* 
    int nn_in_stride = params_cnn.cnn_in_dim[0];
    int nn_out_stride = nn_out_dim[nn_layer_num-1];
    printf("nn_in_stride = %d, nn_out_stride = %d\n", nn_in_stride, nn_out_stride);

    printf( "-------------------------------------\n" );

    // ---------------------------------- //
    // Initial Parameters for NN Training
    // ---------------------------------- //

    long long utter_read_merge = 0;
    long long frames_all_merge = 0;
    float loss_all_merge = 0.0f;
    long long correct_all_merge = 0;

    int gpu_count;
    gpuErrCheck( cudaGetDeviceCount( &gpu_count ) );

    if( num_threads > 0 && num_threads < gpu_count )
	gpu_count= num_threads;
    // gpu_count = 1;

    pthread_t *train_thread = (pthread_t *)malloc( sizeof(pthread_t) * gpu_count );
    lbx_pthread_arg *thread_arg = (lbx_pthread_arg *)malloc( sizeof(lbx_pthread_arg) * gpu_count );

    int thread_count;
    int pth_func_return;

    for( thread_count = 0; thread_count< gpu_count; thread_count++ ) {

	thread_arg[thread_count].gpu_id_s = thread_count;

	thread_arg[thread_count].cv_flag_s = crossvalidate;

	thread_arg[thread_count].nn_para_cpu_s = nn_para_cpu;
	thread_arg[thread_count].nn_layer_num_s = nn_layer_num;
	thread_arg[thread_count].nn_para_num_s = nn_para_num;


	thread_arg[thread_count].cnn_cpu_s = cnn_cpu;
	thread_arg[thread_count].params_cnn_s = &params_cnn;
	thread_arg[thread_count].cnn_hidden_node_num_s= cnn_hidden_node_num;

	thread_arg[thread_count].nn_in_dim_s = nn_in_dim;
	thread_arg[thread_count].nn_out_dim_s = nn_out_dim;

	thread_arg[thread_count].frame_length_s = feature_in_dim;
	thread_arg[thread_count].splice_length_s = splice_length;
	thread_arg[thread_count].splice_index_s = splice_index; 

	thread_arg[thread_count].nn_in_stride_s = nn_in_stride;
	thread_arg[thread_count].nn_out_stride_s = nn_out_stride;

	thread_arg[thread_count].learn_rate_s = learn_rate;
	thread_arg[thread_count].batch_size_s = bunchsize;
	thread_arg[thread_count].updates = num_updates;
	thread_arg[thread_count].train_data_buffer_size_s = train_data_buffer_size;

	thread_arg[thread_count].add_row_data_cpu_s = add_row_data;
	thread_arg[thread_count].mul_row_data_cpu_s = mul_row_data;

	thread_arg[thread_count].utter_all_merge_s = &utter_read_merge;
	thread_arg[thread_count].frames_all_merge_s = &frames_all_merge;
	thread_arg[thread_count].loss_all_merge_s = &loss_all_merge;
	thread_arg[thread_count].correct_all_merge_s = &correct_all_merge;
	thread_arg[thread_count].multicard = (gpu_count > 1);

    }

    finished_reading_a_utter = true;

    start_time = time(NULL);

    printf( "       Parellel Training Start!\n" );
    printf( "-------------------------------------\n" );
    fflush(stdout);

    // ######## Parallel Training Begin ########

    for( thread_count = 0; thread_count< gpu_count; thread_count++ ) {

	pth_func_return = pthread_create( &train_thread[thread_count], NULL, lbxParallelTrainLock, &thread_arg[thread_count] );

	if( pth_func_return ) {
	    printf( "Thread %d Create Error!\n", thread_count);
	    return -4;
	}
    }

    void *pth_return;

    for( thread_count = 0; thread_count< gpu_count; thread_count++ ) {
	pth_func_return = pthread_join( train_thread[thread_count], &pth_return );

	if( pth_func_return || pth_return != ((void *)0) ) {
	    printf( "Thread %d Join Error!\n", thread_count);
	    return -5;
	}
    }	

    finish_time = time(NULL);

    total_time = finish_time - start_time;

    printf( "-------------------------------------\n" );
    printf( "             Finish!\n" );
    printf( "-------------------------------------\n" );

    fclose(file_pointer_key);

    free(train_thread);
    free(thread_arg);

    free(nn_in_dim);
    free(nn_out_dim);

    free(nn_para_cpu);
    free(cnn_cpu);

    return 0;
}

void *lbxParallelTrainLock ( void *arg ) {

    lbx_pthread_arg *thread_arg_t = (lbx_pthread_arg *) arg;

    // printf("    Thread %d Create Sucessfully\n", thread_arg_t->gpu_id_s );

    time_t start_time, finish_time;
    long long total_time;
    long long cnn_back_time,cnn_pro_time, dnn_back_time, dnn_pro_time;
    time_t point_t,cnn_b_time_t, cnn_p_time_t, dnn_b_time_t, dnn_p_time_t;

    bool cv_flag = thread_arg_t->cv_flag_s;

    bool multicard = thread_arg_t->multicard;

    float *nn_para_cpu = thread_arg_t->nn_para_cpu_s;

    int active_GPU_ID = thread_arg_t->gpu_id_s;
    gpuErrCheck( cudaSetDevice( active_GPU_ID ) );

    int update_freq	= thread_arg_t->updates;
    int update_count = 0;

    float learn_rate = thread_arg_t->learn_rate_s;
    int batch_size = thread_arg_t->batch_size_s;
    int train_data_buffer_size = thread_arg_t->train_data_buffer_size_s;

    int layer_count;

    int frame_length = thread_arg_t->frame_length_s;
    int	splice_length = thread_arg_t->splice_length_s;

    int *splice_index = (int *)malloc( splice_length * sizeof(int) );

    ///*********CNN*****************
    float *cnn_cpu = thread_arg_t->cnn_cpu_s;
    cnn_params *params_cnn = thread_arg_t->params_cnn_s;
    int cnn_hidden_node_num = thread_arg_t->cnn_hidden_node_num_s;

    // printf("splice_index: ");
    for ( layer_count = 0; layer_count < splice_length; layer_count++ ) {
	splice_index[layer_count] = (thread_arg_t->splice_index_s)[layer_count]; 
	// printf("%d ", splice_index[layer_count]);
    }
    // printf("\n");

    int pre_frame_MAX = 0, post_frame_MAX = 0;
    int pre_frame_LENGTH = 0, post_frame_LENGTH = 0;

    for (int i = 0; i < splice_length; ++i) {

	if( splice_index[i] < 0 ) {

	    pre_frame_LENGTH++;

	    if( (-splice_index[i]) > pre_frame_MAX )
		pre_frame_MAX = -splice_index[i];
	} 
	else if ( splice_index[i] > 0 ) {

	    post_frame_LENGTH++;

	    if( splice_index[i] > post_frame_MAX )
		post_frame_MAX = splice_index[i];
	}

    }
    // printf("PRE_FRAME_MAX = %d, POST_FRAME_MAX = %d\n", pre_frame_MAX, post_frame_MAX);
    // printf("PRE_FRAME_LENGTH = %d, POST_FRAME_LENGTH = %d\n", pre_frame_LENGTH, post_frame_LENGTH);

    int nn_layer_num = thread_arg_t->nn_layer_num_s;

    int nn_in_stride = thread_arg_t->nn_in_stride_s;
    int nn_out_stride = thread_arg_t->nn_out_stride_s;

    int *nn_in_dim = (int *)malloc(nn_layer_num * sizeof(int));
    int *nn_out_dim = (int *)malloc(nn_layer_num * sizeof(int));

    for ( layer_count = 0; layer_count<nn_layer_num; layer_count++ ) {
	nn_in_dim[layer_count] = (thread_arg_t->nn_in_dim_s)[layer_count];
	nn_out_dim[layer_count] = (thread_arg_t->nn_out_dim_s)[layer_count];
    }

    int *nn_weight_pointer_stride = (int *)malloc( nn_layer_num * sizeof(int) );
    int *nn_bias_pointer_stride = (int *)malloc( nn_layer_num * sizeof(int) );
    int *nn_result_pointer_stride = (int *)malloc( nn_layer_num * sizeof(int) );

    nn_weight_pointer_stride[0]	= 0;
    nn_bias_pointer_stride[0] = nn_in_dim[0] * nn_out_dim[0];
    nn_result_pointer_stride[0] = 0;

    int nn_hidden_node_num = 0;
    int nn_para_num = ( nn_in_dim[nn_layer_num-1] +1 )* nn_out_dim[nn_layer_num-1];

    for (layer_count = 0; layer_count<(nn_layer_num-1); layer_count++) {
	nn_hidden_node_num += nn_out_dim[layer_count];
	nn_para_num += ( nn_in_dim[layer_count] +1 )* nn_out_dim[layer_count];
    }
    // std::cout<< "Hidden Neuron Num = " << nn_hidden_node_num << std::endl;

    for ( layer_count=1; layer_count<nn_layer_num; layer_count++ ) {

	nn_weight_pointer_stride[layer_count] = nn_bias_pointer_stride[layer_count-1] +
	    nn_out_dim[layer_count-1];

	nn_bias_pointer_stride[layer_count] = nn_weight_pointer_stride[layer_count] + 
	    ( nn_in_dim[layer_count] * nn_out_dim[layer_count] );

	nn_result_pointer_stride[layer_count] = nn_result_pointer_stride[layer_count-1] +
	    ( nn_out_dim[layer_count-1] * batch_size );

    }

    if( nn_para_num != thread_arg_t->nn_para_num_s ) {
	printf("NN Para Num Mismatch\n");
	return (void *)(-2);
    }

    // ------------------ //
    // Initialize Program
    // ------------------ //

    // we create (IO_READ_BUF_SIZE - 1) more rows to buffer the block overflow
    float *train_data_cpu_temp = (float *)malloc( (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * nn_in_stride * sizeof(float) );

    int *label_buffer_cpu = (int *)malloc( (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(int) );

    // we don't shuffle the data in the buffer, we just random their indexes
    int *shuffle_index_cpu = (int *)malloc( (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(int) );

    int *tgt_out_index_cpu = (int *)malloc( (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(int) );

    // GPU data & parameters
    float *train_data_buffer_gpu;
    float *temp_train_data_gpu;

    gpuErrCheck( cudaMalloc((void**)&train_data_buffer_gpu, 
		(train_data_buffer_size + IO_READ_BUF_SIZE - 1) * nn_in_stride * sizeof(float)) );
    gpuErrCheck( cudaMemset( train_data_buffer_gpu, 0, 
		(train_data_buffer_size + IO_READ_BUF_SIZE - 1) * nn_in_stride * sizeof(float)) );

    /// modified by MHZ
    //********************************************************** 
    float *train_data_temp_gpu;
    gpuErrCheck( cudaMalloc((void**)&train_data_temp_gpu, 
		(train_data_buffer_size + IO_READ_BUF_SIZE - 1) * nn_in_stride * sizeof(float)) );
    //********************************************************** 
    // we need the PRE/POST FRAMES to fill in the blocks  
    gpuErrCheck( cudaMalloc((void**)&temp_train_data_gpu, (IO_READ_BUF_SIZE + pre_frame_MAX + post_frame_MAX) * frame_length * sizeof(float)) );

    int *splice_index_gpu;
    float *add_row_data_gpu;
    float *mul_row_data_gpu;

    cudaMalloc((void**)&splice_index_gpu, splice_length * sizeof(float));

    cudaMalloc((void**)&add_row_data_gpu, frame_length * splice_length * sizeof(float));

    cudaMalloc((void**)&mul_row_data_gpu, frame_length * splice_length * sizeof(float));

    gpuErrCheck( cudaMemcpy( splice_index_gpu,
		splice_index,
		splice_length * sizeof(float),
		cudaMemcpyHostToDevice ) );

    gpuErrCheck( cudaMemcpy( add_row_data_gpu,
		thread_arg_t->add_row_data_cpu_s,
		frame_length * splice_length * sizeof(float),
		cudaMemcpyHostToDevice ) );

    gpuErrCheck( cudaMemcpy( mul_row_data_gpu,
		thread_arg_t->mul_row_data_cpu_s,
		frame_length * splice_length * sizeof(float),
		cudaMemcpyHostToDevice ) );

    // ---------------------------------- //
    // Initial Parameters for NN Training
    // ---------------------------------- //

    int *correct_gpu;
    float *xent_gpu;

    gpuErrCheck( cudaMalloc( (void**)&correct_gpu, sizeof(int) ) );
    gpuErrCheck( cudaMalloc( (void**)&xent_gpu, sizeof(float) ) );

    // ####### GPU Initialization #######
    // CUDA & cuBLAS state parameters

    float *nn_para_gpu;				// Weight on GPU
    float *nn_para_gpu_BAK;
    float *nn_para_delta = (float *)malloc( nn_para_num * sizeof(float) );

    float *nn_in_gpu, *nn_out_gpu;

    int *tgt_out_index_gpu;
    int *shuffle_index_gpu;

    float *nn_diff_gpu;
    float *nn_hidden_result_gpu;	// Neuron results, expect the input layer
    float *nn_delta_gpu;			// Delta for BP 

    gpuErrCheck( cudaMalloc((void**)&nn_para_gpu, nn_para_num * sizeof(float)) );

    gpuErrCheck( cudaMalloc((void**)&nn_para_gpu_BAK, nn_para_num * sizeof(float)) );

    gpuErrCheck( cudaMalloc((void**)&nn_in_gpu, nn_in_stride * (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(float)) );
    gpuErrCheck( cudaMemset( nn_in_gpu, 0, 
		nn_in_stride * (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(float) ) );

    gpuErrCheck( cudaMalloc((void**)&nn_out_gpu, nn_out_stride * batch_size * sizeof(float)) );
    gpuErrCheck( cudaMemset( nn_out_gpu, 0, nn_out_stride * batch_size * sizeof(float) ) );

    gpuErrCheck( cudaMalloc((void**)&tgt_out_index_gpu, (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(float)) );

    gpuErrCheck( cudaMalloc((void**)&shuffle_index_gpu, (train_data_buffer_size + IO_READ_BUF_SIZE - 1) * sizeof(float)) );

    gpuErrCheck( cudaMalloc((void**)&nn_diff_gpu, nn_out_stride * batch_size * sizeof(float)) );

    gpuErrCheck( cudaMalloc((void**)&nn_hidden_result_gpu, nn_hidden_node_num * batch_size * sizeof(float)) );

    gpuErrCheck( cudaMalloc((void**)&nn_delta_gpu, nn_hidden_node_num * batch_size * sizeof(float)) );

    gpuErrCheck( cudaMemcpy( nn_para_gpu,
		nn_para_cpu,
		nn_para_num * sizeof(float),
		cudaMemcpyHostToDevice ) );

    /// CNN Part
    //************************************************
    float *cnn_para_delta;


    float *cnn_para_gpu_BAK;
    float *cnn_para_gpu, *cnn_para_grad_gpu;
    float *cnn_delta_gpu;
    float *cnn_hidden_result_gpu;

    float *mid_data_gpu;
    float *mid_delta_gpu;
    int mid_data_stride = nn_in_dim[0];
    int mid_delta_stride = nn_in_dim[0];

    int *mask; /// mask saves the pooling indices

    float *transform_matrix; /// for feature transform

    cnn_para_delta = (float *)malloc(params_cnn->cnn_para_num * sizeof(float));

    gpuErrCheck( cudaMalloc((void**)&mid_data_gpu, nn_in_dim[0] * batch_size*sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&mid_delta_gpu, nn_in_dim[0] * batch_size*sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&cnn_hidden_result_gpu, cnn_hidden_node_num*batch_size*sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&cnn_para_gpu, params_cnn->cnn_para_num * sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&cnn_para_grad_gpu, params_cnn->cnn_para_num * sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&cnn_para_gpu_BAK, params_cnn->cnn_para_num * sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&cnn_delta_gpu, cnn_hidden_node_num * batch_size*sizeof(float)));
    gpuErrCheck( cudaMalloc((void**)&mask, params_cnn->pool_out_dim[0]*batch_size*sizeof(int)));
    gpuErrCheck( cudaMalloc((void**)&transform_matrix, params_cnn->cnn_in_dim[0] * params_cnn->cnn_in_dim[0] * sizeof(float)));

    gpuErrCheck( cudaMemset(mid_data_gpu, 0, nn_in_dim[0] * batch_size*sizeof(float) ));
    gpuErrCheck( cudaMemset( mid_delta_gpu, 0, nn_in_dim[0] * batch_size*sizeof(float)));
    gpuErrCheck( cudaMemset( cnn_hidden_result_gpu, 0, cnn_hidden_node_num*batch_size*sizeof(float)));
    gpuErrCheck( cudaMemset( cnn_delta_gpu, 0, cnn_hidden_node_num * batch_size*sizeof(float)));
    gpuErrCheck( cudaMemset( mask, 0, params_cnn->pool_out_dim[0]*batch_size*sizeof(int)));

    gpuErrCheck( cudaMemcpy( cnn_para_gpu,
		cnn_cpu,
		params_cnn->cnn_para_num * sizeof(float),
		cudaMemcpyHostToDevice ) );

    cnnSetTransMatrix(transform_matrix, params_cnn->cnn_in_dim[0], params_cnn->patch_stride[0]);

    //******************************************************************
    // assistant all-one vector
    float *vec_one_gpu; 		// All 1-vector, length = batch_size
    gpuErrCheck( cudaMalloc((void**)&vec_one_gpu, batch_size * sizeof(float)) );
    lbxSetConstGPU( batch_size, vec_one_gpu, 1.0f);

    float *train_data_buffer_cpu;
    float *nn_out_cpu;
    train_data_buffer_cpu = (float *)malloc((train_data_buffer_size + IO_READ_BUF_SIZE - 1) * nn_in_stride * sizeof(float));
    nn_out_cpu = (float *)malloc(nn_out_stride * batch_size * sizeof(float));


    int size_row_filled_in_buffer = 0;
    int size_row_filled_in_buffer_last_time = 0;
    int index_top_of_the_buffer = 0;

    long long int utter_count = 0;
    long long int readin_size = 0;
    int utter_progress = 0;

    float loss_all = 0;
    long long correct_all = 0, frames_all = 0;
    int frames_progress = 0;
    float loss_progress = 0;

    int lbx_func_return_value = 0;

    printf( "-------------------------------------\n" );
    printf( "             GPU %d Start!\n", active_GPU_ID );
    printf( "-------------------------------------\n" );

    printf("%d %d %d\n ",nn_in_dim[0],params_cnn->pool_out_dim[0], mid_data_stride);

    fflush(stdout);
    start_time = time(NULL);

    update_count = 0;

    while ( total_size > 0 || size_row_filled_in_buffer > 0) {

	// ----------------------- //
	// begin filling a buffer
	// ----------------------- //

	// ############################
	// begin mutex for reading training data
	pthread_mutex_lock( &train_data_mutex );
	if (total_size == 0 && size_row_filled_in_buffer == 0)
	{
	    pthread_mutex_unlock( &train_data_mutex );
	    return (void *)(0);
	}

	if (total_size > 0) {
	    if (total_size < train_data_buffer_size)
	    {
		readin_size = total_size ;
	    }
	    else {
		readin_size = train_data_buffer_size;
	    }
	    fread(train_data_buffer_cpu, sizeof(float), readin_size * nn_in_stride, file_pointer_key);
	    cudaMemcpy(train_data_buffer_gpu, train_data_buffer_cpu, sizeof(float) * readin_size * nn_in_stride, cudaMemcpyHostToDevice);
	    utter_count += readin_size;
	    utter_progress += readin_size;
	    size_row_filled_in_buffer += readin_size;
	    total_size -= readin_size;

	    printf("total size:%ld readin_size:%d\n",total_size, readin_size);
	    fflush(stdout);

	    // end mutex for reading training data
	    // #--------------------------#

	    // printf( "GPU[%d], row_size_in_buffer = %d\n", active_GPU_ID, size_row_filled_in_buffer );



	    //Feature Transform for CNN
	    //***************************************************************

	    cnnTransform(size_row_filled_in_buffer - size_row_filled_in_buffer_last_time, params_cnn->cnn_in_dim[0], nn_in_stride,
		    transform_matrix, train_data_buffer_gpu + size_row_filled_in_buffer_last_time * nn_in_stride, train_data_temp_gpu);
	    /*
	       float *temp = train_data_buffer_gpu;
	       train_data_buffer_gpu = train_data_temp_gpu;
	       train_data_temp_gpu = temp;
	     */
	    //***************************************************************
	}
	pthread_mutex_unlock( &train_data_mutex );
	if( size_row_filled_in_buffer < batch_size ) {
	    batch_size = size_row_filled_in_buffer;
	}
	if( !cv_flag ) {

	    // ----------------------------------- //
	    // shuffle the buffer (the data index)
	    // ----------------------------------- //

	    for (int i = 0; i < (train_data_buffer_size + IO_READ_BUF_SIZE - 1); i++ ) {
		shuffle_index_cpu[i] = i;
	    }

	    ShuffleArray_Fisher_Yates( shuffle_index_cpu, size_row_filled_in_buffer );

	    for (int i = 0; i < size_row_filled_in_buffer; i++ ) {
		tgt_out_index_cpu[i] = label_buffer_cpu[ shuffle_index_cpu[i] ];
	    }

	    /*
	       FILE *train_data_file;
	       train_data_file = fopen("index.dat","w");
	       for (int i = 0; i < size_row_filled_in_buffer; i++)
	       fprintf(train_data_file, "%d\n", shuffle_index_cpu[i]);
	       fclose(train_data_file);
	     */
	    gpuErrCheck( cudaMemcpy( shuffle_index_gpu,
			shuffle_index_cpu,
			size_row_filled_in_buffer * sizeof(int),
			cudaMemcpyHostToDevice ) );

	    lbxPrefetchDataAccordingToShuffleIndexGPU( size_row_filled_in_buffer, params_cnn->cnn_in_dim[0], nn_in_stride, 
		    shuffle_index_gpu, 
		    train_data_buffer_gpu, nn_in_gpu );

	} else {

	    gpuErrCheck( cudaMemcpy( nn_in_gpu,
			train_data_buffer_gpu,
			size_row_filled_in_buffer * nn_in_stride * sizeof(float),
			cudaMemcpyDeviceToDevice ) );

	    memcpy( tgt_out_index_cpu, 
		    label_buffer_cpu, 
		    size_row_filled_in_buffer * sizeof(int) );
	}

	gpuErrCheck( cudaMemcpy( tgt_out_index_gpu,
		    tgt_out_index_cpu,
		    size_row_filled_in_buffer * sizeof(int),
		    cudaMemcpyHostToDevice ) );

	index_top_of_the_buffer = 0;

	while ( size_row_filled_in_buffer >= batch_size ) {


	    update_count++;
	    // forward pass
	    cnnNetPropa(cnn_para_gpu, cnn_hidden_result_gpu,
		    nn_in_gpu + (index_top_of_the_buffer * nn_in_stride), nn_in_stride,
		    mid_data_gpu, mid_data_stride,
		    params_cnn, batch_size,
		    vec_one_gpu, mask);

	    lbxNetPropagation( nn_para_gpu, nn_hidden_result_gpu, 
		    mid_data_gpu, nn_out_gpu,
		    nn_layer_num, batch_size, 
		    nn_in_dim, nn_out_dim, 
		    mid_data_stride, nn_out_stride, 
		    nn_weight_pointer_stride, nn_bias_pointer_stride, nn_result_pointer_stride, 
		    vec_one_gpu );	


	    cudaMemcpy(nn_out_cpu, nn_out_gpu, sizeof(float) * nn_out_stride * batch_size, cudaMemcpyDeviceToHost);
	pthread_mutex_lock( &train_data_mutex );
	    fwrite(nn_out_cpu, sizeof(float), nn_out_stride * batch_size, file_out_pointer);
	pthread_mutex_unlock( &train_data_mutex );


	    index_top_of_the_buffer += batch_size;
	    size_row_filled_in_buffer -= batch_size;			

	}

	size_row_filled_in_buffer_last_time = size_row_filled_in_buffer;

	if( size_row_filled_in_buffer > 0 ) {

	    // printf("finish = %d, size = %d, index = %d\n", finished_reading_a_utter, size_row_filled_in_buffer, index_top_of_the_buffer);

	    gpuErrCheck( cudaMemcpy( train_data_buffer_gpu,
			nn_in_gpu + (index_top_of_the_buffer * nn_in_stride),
			size_row_filled_in_buffer * nn_in_stride * sizeof(float),
			cudaMemcpyDeviceToDevice ) );

	    // copy the overflow label up to the top
	    memcpy( label_buffer_cpu, 
		    tgt_out_index_cpu + index_top_of_the_buffer, 
		    size_row_filled_in_buffer * sizeof(int) );

	}

	if( utter_progress >5000 ) {

	    finish_time = time(NULL);

	    total_time = finish_time - start_time;

	    printf( "GPU [%d]: After %lld utterances: time elapsed = %2.3f min; processed %5.2lf frames per second\n",
		    active_GPU_ID, utter_count, total_time/60.0, (double)frames_all/total_time );

	    fflush(stdout);

	    utter_progress = 0;
	}


    }

    finish_time = time(NULL);

    total_time = finish_time - start_time;

    printf( "-------------------------------------\n" );
    printf( "GPU [%d] Finish Training! After %lld utterances: time elapsed = %2.3f min; processed %5.2lf frames per second\n",
	    active_GPU_ID, utter_count, total_time/60.0, (double)frames_all/total_time );
    printf( "-------------------------------------\n" );

    fflush(stdout);



    // ############################
    pthread_mutex_lock( &xent_mutex );

    *(thread_arg_t->utter_all_merge_s) += utter_count;
    *(thread_arg_t->frames_all_merge_s) += frames_all;

    *(thread_arg_t->loss_all_merge_s) += loss_all;
    *(thread_arg_t->correct_all_merge_s) += correct_all;

    pthread_mutex_unlock( &xent_mutex );
    // #--------------------------#


    cudaFree(train_data_buffer_gpu);
    cudaFree(temp_train_data_gpu);

    cudaFree(add_row_data_gpu);
    cudaFree(mul_row_data_gpu);

    cudaFree(correct_gpu);
    cudaFree(xent_gpu);

    cudaFree(nn_para_gpu);
    cudaFree(nn_para_gpu_BAK);

    cudaFree(nn_in_gpu);

    cudaFree(nn_out_gpu);

    cudaFree(tgt_out_index_gpu);
    cudaFree(shuffle_index_gpu);

    cudaFree(nn_diff_gpu);
    cudaFree(nn_hidden_result_gpu);
    cudaFree(nn_delta_gpu);			

    // CNN

    cudaFree(mid_data_gpu);
    cudaFree(mid_delta_gpu);
    cudaFree(cnn_para_gpu);
    cudaFree(cnn_hidden_result_gpu);
    cudaFree(cnn_delta_gpu);
    cudaFree(mask);
    cudaFree(transform_matrix);
    cudaFree(train_data_temp_gpu);

    free(nn_in_dim);
    free(nn_out_dim);

    free( nn_para_delta );

    free(nn_weight_pointer_stride);
    free(nn_bias_pointer_stride);
    free(nn_result_pointer_stride);

    free(train_data_cpu_temp);

    free(label_buffer_cpu);
    free(shuffle_index_cpu);

    free(tgt_out_index_cpu);


    return (void *)(0);
}
