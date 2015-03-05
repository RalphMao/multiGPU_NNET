#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cublas.h>
#include <cuda_runtime_api.h>

#include "gpuErrCheck_INLINE_H_.h"
#include "lbx_cuda_kernels.h"

#include "lbx_nn.h"

void lbxNetEval( float *nn_out_gpu, int *tgt_out_index_gpu, 
	float *diff_gpu, 
    int nn_out_dim, int batch_size, 
    int nn_out_stride,
    int *correct_gpu, float *xent_gpu,
    float *loss_all,
    long long *correct_all, 
    long long * frames_all,
    int *frames_progress,
    float *loss_progress ) {
  
	// calculate diff for BP
	lbxNNDiffGPU( nn_out_dim, batch_size, nn_out_stride, nn_out_gpu, tgt_out_index_gpu, diff_gpu );

	// calculate correct
    int correct = 0;

    lbxAccuracyEvalGPU( nn_out_dim, batch_size, nn_out_stride, 
		nn_out_gpu, tgt_out_index_gpu, correct_gpu );

    gpuErrCheck( cudaMemcpy( &correct, correct_gpu, sizeof(int), cudaMemcpyDeviceToHost ) );

    float cross_entropy = 0;

    lbxXentEvalGPU( batch_size, nn_out_stride, nn_out_gpu, tgt_out_index_gpu, xent_gpu );

    gpuErrCheck( cudaMemcpy( &cross_entropy, xent_gpu, sizeof(float), cudaMemcpyDeviceToHost ) );
	
	// printf("Correct = %d, Xent = %f\n", correct, cross_entropy );

    // accumulate
    (*loss_all) += cross_entropy;
    (*correct_all) += correct;
    (*frames_all) += batch_size;

    // progressive loss reporting
    (*frames_progress) += batch_size;
    (*loss_progress) += cross_entropy;
    
    if ( (*frames_progress) > progress_step) {

    	int active_GPU_ID;
    	gpuErrCheck( cudaGetDevice( &active_GPU_ID ) );

        printf( "GPU[%d] ProgressLoss[%dh/%dh]: Xent = %.4lf, Accuracy = %2.2lf%%\n", 
        	active_GPU_ID,
        	(*frames_progress)/100/3600,
        	(*frames_all)/100/3600, 
        	(*loss_progress)/ (*frames_progress),
        	(double)(*correct_all) / (double)(*frames_all) * 100 );
			
		fflush(stdout);
        
        *frames_progress = 0;
        *loss_progress = 0.0f;
    
    }
}

void lbxNetPropagation( float *nn_para_gpu, float *nn_hidden_result_gpu, 
	float *train_data, float *nn_out_gpu,   
	int layer_num, int batch_size, 
	int *nn_in_dim, int *nn_out_dim, 
	int train_data_stride, int nn_out_stride, 
	int *weight_stride, int *bias_stride, int *result_stride, 
	float *vec_one_gpu ) {

	int layer_count;

	if( layer_num < 2 ) {
		printf("Layer Num = %d, Please Contact the Designers...\n", layer_num);
		return;
	}

    cublasSgemm( 'N', 'N',
		nn_out_dim[0], batch_size, 1,  
		1.0f,
		nn_para_gpu + bias_stride[0], nn_out_dim[0],
		vec_one_gpu, 1,
		0,
		nn_hidden_result_gpu, nn_out_dim[0] );

	cublasSgemm( 'T', 'N',
		nn_out_dim[0], batch_size, nn_in_dim[0],  
		1.0f,
		nn_para_gpu, nn_in_dim[0],
		train_data, train_data_stride,
		1.0f,
		nn_hidden_result_gpu, nn_out_dim[0] );

	lbxSigmoidGPU( nn_out_dim[0]*batch_size, nn_hidden_result_gpu, nn_hidden_result_gpu );

	for (layer_count=1; layer_count<(layer_num-1); layer_count++) {

		cublasSgemm( 'N', 'N',
			nn_out_dim[layer_count], batch_size, 1,  
			1.0f,
			nn_para_gpu + bias_stride[layer_count], nn_out_dim[layer_count],
			vec_one_gpu, 1,
			0,
			nn_hidden_result_gpu + result_stride[layer_count], nn_out_dim[layer_count] );

		cublasSgemm( 'T', 'N',
			nn_out_dim[layer_count], batch_size, nn_in_dim[layer_count],  
			1.0f,
			nn_para_gpu + weight_stride[layer_count], nn_in_dim[layer_count],
			nn_hidden_result_gpu + result_stride[layer_count-1], nn_in_dim[layer_count],
			1.0f,
			nn_hidden_result_gpu + result_stride[layer_count], nn_out_dim[layer_count] );

		lbxSigmoidGPU( nn_out_dim[layer_count]*batch_size, 
			nn_hidden_result_gpu + result_stride[layer_count], 
			nn_hidden_result_gpu + result_stride[layer_count] );
	gpuErrCheck( cudaPeekAtLastError() );

	}
	
	cublasSgemm( 'N', 'N',
		nn_out_dim[layer_num-1], batch_size, 1,  
		1.0f,
		nn_para_gpu + bias_stride[layer_num-1], nn_out_dim[layer_num-1],
		vec_one_gpu, 1,
		0,
		nn_out_gpu, nn_out_stride );

	cublasSgemm( 'T', 'N',
		nn_out_dim[layer_num-1], batch_size, nn_in_dim[layer_num-1],  
		1.0f,
		nn_para_gpu + weight_stride[layer_num-1], nn_in_dim[layer_num-1],
		nn_hidden_result_gpu + result_stride[layer_num-2], nn_in_dim[layer_num-1],
		1.0f,
		nn_out_gpu, nn_out_stride );

	lbxSoftmaxGPU( nn_out_dim[layer_num-1], batch_size, nn_out_stride, nn_out_gpu, nn_out_gpu );

	gpuErrCheck( cudaPeekAtLastError() );

}


void lbxNetBackProp( float *nn_para_gpu, 
	float *nn_hidden_result_gpu, float *nn_delta_gpu,
	float *train_data, float *obj_diff,   
	int layer_num, int batch_size, 
	int *nn_in_dim, int *nn_out_dim, 
	int train_data_stride, int obj_diff_stride, 
	int *weight_stride, int *bias_stride, int *result_stride, 
	float *mid_delta_gpu, int mid_delta_stride, //new
	float learn_rate, 
	float *vec_one_gpu ) {

	int layer_count;

	if( layer_num < 2 ) {
		printf("Layer Num = %d, Really Need BP?\n", layer_num);
		return;
	}

    cublasSgemm( 'N', 'N',
		nn_in_dim[layer_num-1], batch_size, nn_out_dim[layer_num-1],  
		1.0f,
		nn_para_gpu + weight_stride[layer_num-1], nn_in_dim[layer_num-1],
		obj_diff, obj_diff_stride,
		0,
		nn_delta_gpu + result_stride[layer_num-2], nn_in_dim[layer_num-1] );
		
	lbxDiffSigmoidGPU( nn_out_dim[layer_num-2]*batch_size, 
		nn_delta_gpu + result_stride[layer_num-2], 
		nn_hidden_result_gpu + result_stride[layer_num-2], 
		nn_delta_gpu + result_stride[layer_num-2] );
	
	for ( layer_count=(layer_num-3); layer_count>=0; layer_count-- ) {
	
		cublasSgemm( 'N', 'N',
			nn_in_dim[layer_count+1], batch_size, nn_out_dim[layer_count+1],  
			1.0f,
			nn_para_gpu + weight_stride[layer_count+1], nn_in_dim[layer_count+1],
			nn_delta_gpu + result_stride[layer_count+1], nn_out_dim[layer_count+1],
			0,
			nn_delta_gpu + result_stride[layer_count], nn_in_dim[layer_count+1] );

		lbxDiffSigmoidGPU( nn_out_dim[layer_count]*batch_size, 
			nn_delta_gpu + result_stride[layer_count], 
			nn_hidden_result_gpu + result_stride[layer_count], 
			nn_delta_gpu + result_stride[layer_count] );
	}

    // Tail delta saved for CNN backpropagation
    cublasSgemm( 'N', 'N',
	    nn_in_dim[0], batch_size, nn_out_dim[0],  
	    1.0f,
	    nn_para_gpu + weight_stride[0], nn_in_dim[0],
	    nn_delta_gpu + result_stride[0], nn_out_dim[0],
	    0.0f,
	    mid_delta_gpu,mid_delta_stride );

	// Update weight

	cublasSgemm( 'N', 'T',
		nn_in_dim[layer_num-1], nn_out_dim[layer_num-1], batch_size,
		-learn_rate,
		nn_hidden_result_gpu + result_stride[layer_num-2], nn_in_dim[layer_num-1],
		obj_diff, obj_diff_stride,
		1.0f,
		nn_para_gpu + weight_stride[layer_num-1], nn_in_dim[layer_num-1] );
		
	for( layer_count=(layer_num-2); layer_count>0; layer_count-- ) {
		
		cublasSgemm( 'N', 'T',
			nn_in_dim[layer_count], nn_out_dim[layer_count], batch_size,
			-learn_rate,
			nn_hidden_result_gpu + result_stride[layer_count-1], nn_in_dim[layer_count],
			nn_delta_gpu + result_stride[layer_count], nn_out_dim[layer_count],
			1.0f,
			nn_para_gpu + weight_stride[layer_count], nn_in_dim[layer_count] );	
	}
	
	cublasSgemm( 'N', 'T',
		nn_in_dim[0], nn_out_dim[0], batch_size,
		-learn_rate,
		train_data, train_data_stride,
		nn_delta_gpu + result_stride[0], nn_out_dim[0],
		1.0f,
		nn_para_gpu, nn_in_dim[0] );

	// Update bias

	cublasSgemv( 'N',
		nn_out_dim[layer_num-1], batch_size,
		-learn_rate,
		obj_diff, obj_diff_stride,
		vec_one_gpu, 1,
		1.0f,
		nn_para_gpu + bias_stride[layer_num-1], 1 );

	for( layer_count=(layer_num-2); layer_count>=0; layer_count-- ) {
		
		cublasSgemv( 'N',
			nn_out_dim[layer_count], batch_size,
			-learn_rate,
			nn_delta_gpu + result_stride[layer_count], nn_out_dim[layer_count],
			vec_one_gpu, 1,
			1.0f,
			nn_para_gpu + bias_stride[layer_count], 1 );
		
	}

	gpuErrCheck( cudaPeekAtLastError() );
	
}
