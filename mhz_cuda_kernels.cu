#include<cuda_runtime_api.h>
#include <stdio.h>
#include <cuda.h>

inline int n_blocks(int size, int block_size) { 
  return size / block_size + ((size % block_size == 0)? 0 : 1); 
}

__global__ void _max_stride(float *src,float *dst, int stride, int src_ldx, int dst_ldx, int step, int size, int *mask)
{
    int i,r;
    r = blockDim.x*blockIdx.z+threadIdx.x;
    int dst_id = dst_ldx*blockIdx.x+stride*blockIdx.y+r;
    int src_id = src_ldx*blockIdx.x+stride*step*blockIdx.y+r;
    if (r<stride)
    {
	dst[dst_id] = src[src_id];
	mask[dst_id] = src_id;
	for (i=1;i<size;i++)
	    if (dst[dst_id] < src[src_id+stride*i])
	    {
		mask[dst_id] = src_id+stride*i;
		dst[dst_id] = src[src_id+stride*i];
	    }
    }
}

__global__ void _back_max_stride(float *src_diff, float *dst_diff, int stride, int dst_ldx, int *mask)
{
    int r;
    r = blockDim.x*blockIdx.z+threadIdx.x;
    int dst_id = dst_ldx*blockIdx.x+stride*blockIdx.y+r;

    if (r<stride && mask[dst_id] >= 0)
	src_diff[mask[dst_id]] += dst_diff[dst_id];
}
__global__
static void _max(float* mat, const float* A, int dst_rows, int dst_cols, int dst_stride, int src_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int dst_index = i + j*dst_stride, src_index = i + j*src_stride;
  if ( i < dst_cols  &&  j < dst_rows ) {
    float a = mat[dst_index], b = A[src_index];
    mat[dst_index] = (a > b ? a : b);
  }
} 

extern "C" void max_stride(float* src, float*dst, int stride, int src_ldx, int dst_ldx, int step, int size,int batch_size,int num_stride, int *mask)
{
    dim3 gridDim(batch_size,num_stride,(stride+31)/32);
    _max_stride<<<gridDim,32>>>(src,dst,stride,src_ldx,dst_ldx,step,size,mask);
}

extern "C" void back_max_stride(float *src_diff, float *dst_diff, int stride, int dst_ldx, int batch_size,int num_stride, int *mask)
{
    dim3 gridDim(batch_size,num_stride,(stride+31)/32);
    _back_max_stride<<<gridDim,32>>>(src_diff,dst_diff,stride,dst_ldx,mask);
}


extern "C" void Mhz_cuda_max(float *out_handle, float* hidden_handle, int num_cols, int num_rows, int out_stride, int hidden_stride)
{
   dim3 dimBlock(16, 16);
    dim3 dimGrid(n_blocks(num_cols, 16), n_blocks(num_rows, 16));
    _max<<<dimGrid, dimBlock>>>(out_handle, hidden_handle,num_rows, num_cols, out_stride, hidden_stride);
}
