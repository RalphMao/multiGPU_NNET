#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void max_stride(float* src, float*dst, int stride, int src_ldx, int dst_ldx, int step, int size,int batch_size,int num_stride, int *mask);

int main()
{
    int i;
    float *x;
    float *x_gpu;
    int *mask;
    int *mask_cpu;

    x = (float *)malloc(sizeof(float) * 50);
    mask_cpu = (int *)malloc(sizeof(int) * 8);
    for (i=0;i<50;i++)
	x[i] = (i*50) % 9;

    cudaMalloc((void**)&x_gpu, sizeof(float) * 50);
    cudaMalloc((void**)&mask, sizeof(int) * 8);
    cudaMemcpy(x_gpu,x,sizeof(float) * 20, cudaMemcpyHostToDevice);

    max_stride(x_gpu,x_gpu+20,2,10,4,2,2,2,2,mask);
    cudaMemcpy(x, x_gpu, sizeof(float) * 28, cudaMemcpyDeviceToHost);
    cudaMemcpy(mask_cpu, mask, sizeof(int) * 8, cudaMemcpyDeviceToHost);

    for (i=0; i<20;i++)
	printf("%f ",x[i]);
    printf("\n");
    for (i=20; i<28;i++)
	printf("%f ",x[i]);
    printf("\n");

    for (i = 0; i < 4; i++)
	printf("%d ",mask_cpu[i]);
    return 0;
}

