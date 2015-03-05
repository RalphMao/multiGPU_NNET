

#include "cublas_v2.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "cnnFunc.h"
#include "gpuErrCheck_INLINE_H_.h"

void cnnNetPropa( float *cnn_para_gpu, float *cnn_hidden_result_gpu,
	float *train_data, int train_data_stride,
	float *cnn_out_gpu, int cnn_out_stride,
	const cnn_params *params, int batch_size,
	float *vec_one_gpu, int *mask) {

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
	printf ("cnnCUBLAS initialization failed\n"); 
	return; 
    }

    int num_splice = params->cnn_in_dim[0] / params->patch_stride[0];
    int num_sections = 1 + (params->patch_stride[0] - params->filter_section[0])/ params->filter_step[0];
    int num_patches = 1 + (params->filter_section[0] - params->patch_dim[0]) / params->patch_step[0];
    int num_filters = params->cnn_out_dim[0] / num_patches;
    int num_filters_sec = num_filters/num_sections;
    int filter_dim = num_splice * params->patch_dim[0];
    int cnn_hidden_stride = params->cnn_out_dim[0];
    int num_pools = 1+ (num_patches - params->pool_size[0])/params->pool_step[0];

    float * filter_handle, *bias_handle, *train_data_handle, *hidden_handle, *out_handle;

    float constant_one_float = 1.0f;
    float constant_zero_float = 0.0f;
    /*
       for (i=0;i< params->layer_num; i++)
       {
       if(i == 0)
       {
       */
    for (int s=0; s<num_sections;s++)
    {
	filter_handle = cnn_para_gpu + s * num_filters_sec * filter_dim;
	bias_handle = cnn_para_gpu + num_filters * filter_dim + s * num_filters_sec;

	for (int p=0; p<num_patches;p++)
	{
	    train_data_handle = train_data + (s*params->filter_step[0]+p*params->patch_step[0])*num_splice;
	    hidden_handle = cnn_hidden_result_gpu + (s*num_patches+p) * num_filters_sec;
	    stat = cublasSgemm(handle, CUBLAS_OP_T,CUBLAS_OP_N,
		    num_filters_sec,batch_size,filter_dim,
		    &constant_one_float,
		    filter_handle,filter_dim,
		    train_data_handle, train_data_stride,
		    &constant_zero_float,
		    hidden_handle, cnn_hidden_stride );

	    if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("cnnCUBLAS EXECUTE failed\n"); 
		return; 
	    }
	    stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
		    num_filters_sec,batch_size,1,
		    &constant_one_float,
		    bias_handle,num_filters_sec,
		    vec_one_gpu,1,
		    &constant_one_float,
		    hidden_handle,cnn_hidden_stride );
	    if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("cnnCUBLAS EXECUTE failed\n"); 
		return; 
	    }
	}

	hidden_handle = cnn_hidden_result_gpu + s * num_patches * num_filters_sec; //???
	out_handle = cnn_out_gpu + s * num_pools * num_filters_sec;
	max_stride(hidden_handle,out_handle,num_filters_sec,cnn_hidden_stride,cnn_out_stride,params->pool_step[0],params->pool_size[0],batch_size,num_pools, mask);
    cudaDeviceSynchronize();
    gpuErrCheck( cudaPeekAtLastError() );

    }
    lbxSigmoidGPU(batch_size*cnn_out_stride,cnn_out_gpu,cnn_out_gpu);
    cublasDestroy(handle);
}
void cnnNetBackPropa(float *cnn_para_gpu, float *cnn_hidden_result_gpu,
	float *cnn_delta_gpu, 
	float *train_data, int train_data_stride,
	float *obj_diff, int obj_diff_stride,
	float *obj_data, int obj_data_stride,
	const cnn_params *params, int batch_size,
	float learn_rate,
	float *vec_one_gpu, int *mask) {

    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
	printf ("cnnCUBLAS initialization failed\n"); 
	return; 
    }

    int num_splice = params->cnn_in_dim[0] / params->patch_stride[0];
    int num_sections = 1 + (params->patch_stride[0] - params->filter_section[0])/ params->filter_step[0];
    int num_patches = 1 + (params->filter_section[0] - params->patch_dim[0]) / params->patch_step[0];
    int num_filters = params->cnn_out_dim[0] / num_patches;
    int num_filters_sec = num_filters/num_sections;
    int filter_dim = num_splice * params->patch_dim[0];
    int cnn_hidden_stride = params->cnn_out_dim[0];
    int num_pools = 1+ (num_patches - params->pool_size[0])/params->pool_step[0];

    float * filter_handle, *bias_handle, *train_data_handle, *delta_handle, *out_handle;

    float constant_one_float = 1.0f;
    float constant_zero_float = 0.0f;
    float lr = -learn_rate;

    lbxDiffSigmoidGPU(obj_diff_stride * batch_size, obj_diff, obj_data, obj_diff);

    for (int s=0; s<num_sections;s++)
    {

	delta_handle = cnn_delta_gpu + s * num_patches * num_filters_sec;
	out_handle = obj_diff + s * num_pools * num_filters_sec;

	back_max_stride(delta_handle,out_handle,num_filters_sec,obj_diff_stride,batch_size,num_pools, mask);
    cudaDeviceSynchronize();
    gpuErrCheck( cudaPeekAtLastError() );

	filter_handle = cnn_para_gpu + s * num_filters_sec * filter_dim;
	bias_handle = cnn_para_gpu + num_filters * filter_dim + s * num_filters_sec;

	// Update Weight and Bias
	for (int p=0; p<num_patches;p++)
	{
	    train_data_handle = train_data + (s*params->filter_step[0]+p*params->patch_step[0])*num_splice;
	    delta_handle = cnn_delta_gpu + (s*num_patches+p) * num_filters_sec;

	    stat = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
		    filter_dim, num_filters_sec, batch_size,
		    &lr,
		    train_data_handle, train_data_stride,
		    delta_handle, cnn_hidden_stride,
		    &constant_one_float,
		    filter_handle, filter_dim );

	    if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("cnnCUBLAS EXECUTE failed\n"); 
		return; 
	    }
	    stat = cublasSgemv(handle,CUBLAS_OP_N,
		    num_filters_sec, batch_size,
		    &lr,
		    delta_handle, cnn_hidden_stride,
		    vec_one_gpu, 1,
		    &constant_one_float,
		    bias_handle, 1);
	    if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("cnnCUBLAS EXECUTE failed\n"); 
		return; 
	    }
	}

    }

    cublasDestroy(handle);

}

void cnnSetTransMatrix(float *transform_matrix, int size, int patch_stride)
{

    if (size % patch_stride != 0)
    {
	printf("CNN Transform Error!\n");
	exit(1);
    }

    int num_patches = size / patch_stride;
    float *matrix_cpu;
    matrix_cpu = (float *)malloc(sizeof(float) * size *size);

    for (int i = 0; i < size; i++)
	for (int j = 0; j < size; j++)
	{
	    if (j == patch_stride * (i % num_patches) + i / num_patches)
		matrix_cpu[i*size + j] = 1.0f;
	    else
		matrix_cpu[i*size + j] = 0.0f;
	}

   
    cudaMemcpy(transform_matrix, matrix_cpu, sizeof(float) * size * size, 
	    cudaMemcpyHostToDevice);

    free(matrix_cpu);

}

void cnnTransform(int size_row, int size_col, int stride,
	float *transform_matrix, float *train_data_gpu, float *train_data_temp_gpu) {


    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) { 
	printf ("cnnCUBLAS initialization failed\n"); 
	return; 
    }
    float constant_one_float = 1.0f;
    float constant_zero_float = 0.0f;


    stat = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,
	    size_col, size_row, size_col,
	    &constant_one_float,
	    transform_matrix, size_col,
	    train_data_gpu, stride,
	    &constant_zero_float,
	    train_data_temp_gpu, stride );

	    if (stat != CUBLAS_STATUS_SUCCESS) { 
		printf ("cnnCUBLAS EXECUTE failed\n"); 
		return; 
	    }

    cublasDestroy(handle);
}
