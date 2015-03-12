
#ifndef CNNFUNC
#define CNNFUNC 1

extern "C" void	lbxSigmoidGPU( int N, float *data_in, float *data_out );
extern "C" void	lbxDiffSigmoidGPU( int N, float *delta_in, float *result_in, float *delta_out );
extern "C" void	lbxSetConstGPU( int N, float *data, float val );

extern "C" void back_max_stride(float *src_diff, float *dst_diff, int stride, int dst_ldx, int batch_size,int num_stride, int *mask);
extern "C" void max_stride(float* src, float*dst, int stride, int src_ldx, int dst_ldx, int step, int size,int batch_size,int num_stride, int *mask);  

typedef struct {
    int layer_num;
    int cnn_para_num;
    int *patch_stride;
    int *patch_dim;
    int *patch_step;

    int *filter_section;
    int *filter_step;

    int *pool_size;
    int *pool_step;
    int *pool_stride;

    int *cnn_in_dim, *cnn_out_dim, *pool_out_dim;
} cnn_params;

void cnnNetPropa( float *cnn_para_gpu, float *cnn_hidden_result_gpu,
	float *train_data, int train_data_stride,
	float *cnn_out_gpu, int cnn_out_stride,
	const cnn_params *params, int batch_size,
	float *vec_one_gpu, int *mask);

void cnnNetBackPropa(float *cnn_para_gpu, float *cnn_hidden_result_gpu,
	float *cnn_delta_gpu, float *cnn_para_grad_gpu, 
	float *train_data, int train_data_stride,
	float *obj_diff, int obj_diff_stride,
	float *obj_data, int obj_data_stride,
	const cnn_params *params, int batch_size,
	float learn_rate,
	float *vec_one_gpu, int *mask);

void cnnSetTransMatrix(float *transform_matrix, int size, int patch_stride);

void cnnTransform(int size_row, int size_col, int stride,
	float *transform_matrix, float  *train_data_gpu, float  *train_data_temp_gpu); 

#endif
