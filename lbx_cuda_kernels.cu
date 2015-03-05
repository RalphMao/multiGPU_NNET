#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpuErrCheck_INLINE_H_.h"

#ifndef CUDA_THREADS_PER_BLOCK
#define CUDA_THREADS_PER_BLOCK 256
#endif

// ----------------------------------- //
// IO kernels
// ----------------------------------- //

// this kernel will require the PRE/POST FRAMES to fill features
__global__ static void SpliceFramesToCreatFeatureKernel( int frame_length, int splice_length, 
	int row_size, int row_stride,
	int *splice_index_gpu, 
	float *data_in, float *data_out, 
	int data_in_index_bias, int data_in_length ) {

	// each block process one row of the data
	int row_index = blockIdx.x;

	int THREADS = blockDim.x;
	if ( row_index >= row_size ) return;

	int steps = (frame_length + THREADS - 1) / THREADS;

	for(int i=0; i<steps; i++) {

		int col_index = threadIdx.x + i*THREADS;

		for( int splice_count = 0; splice_count < splice_length; splice_count++ ) {

			int data_out_index = ( row_index * row_stride ) + ( splice_count * frame_length ) + col_index;

			int data_in_row = row_index + splice_index_gpu[ splice_count ] + data_in_index_bias;

			data_in_row = data_in_row > 0 ? data_in_row : 0;
			data_in_row = data_in_row < data_in_length ? data_in_row : (data_in_length-1); 

			int data_in_index = (data_in_row * frame_length) + col_index;

			data_out[data_out_index] = data_in[data_in_index];
		}
	}

}

extern "C" void	lbxSpliceFramesToCreatFeatureGPU( int frame_length, int splice_length, 
	int row_size, int row_stride,
	int *splice_index_gpu,
	float *data_in, float *data_out,
	int data_in_index_bias, int data_in_length ) {
	
	int ThreadsPerBlock = frame_length > CUDA_THREADS_PER_BLOCK ? CUDA_THREADS_PER_BLOCK : frame_length;
	int BlockNum = row_size;
	
	SpliceFramesToCreatFeatureKernel<<< BlockNum, ThreadsPerBlock >>> ( frame_length, splice_length, 
		row_size, row_stride,
		splice_index_gpu, 
		data_in, data_out, 
		data_in_index_bias, data_in_length );

	gpuErrCheck( cudaPeekAtLastError() );
	
}


// mat = (mat + repmat(add_row, 1, col_size) ) .* repmat( mul_row, 1, col_size )
__global__ static void AddRowToMatrixThenRescaleKernel( int row_size, int col_size, int row_stride, 
	float *add_row_data, float *rescale_data, float *mat_data ) {
	
	int m = blockIdx.x*blockDim.x + threadIdx.x;	// row index
	int n = blockIdx.y*blockDim.y + threadIdx.y;	// col index

	if (  m < row_size && n < col_size ) {

		int mat_data_index = m * row_stride + n;

		int row_data_index = n;

		mat_data[mat_data_index] = ( mat_data[mat_data_index] + add_row_data[row_data_index] ) *  rescale_data[row_data_index];
	}

}

extern "C" void	lbxAddRowToMatrixThenRescaleGPU( int row_size, int col_size, int row_stride, 
	float *add_row_data, float *rescale_data, float *mat_data ) {
	
	dim3 dimBlock(16, 16);
	dim3 dimGrid( (row_size + 15)/16, ( col_size + 15)/16 );
	
	AddRowToMatrixThenRescaleKernel<<< dimGrid, dimBlock >>> ( row_size, col_size, row_stride, add_row_data, rescale_data, mat_data );

	gpuErrCheck( cudaPeekAtLastError() );
	
}


__global__ static void PrefetchDataAccordingToShuffleIndexKernel( int row_size, int col_size, int row_stride, 
	int *shuffle_index, float *train_data_buffer, float *nn_out_gpu ) {
	
	int m = blockIdx.x*blockDim.x + threadIdx.x;	// row index
	int n = blockIdx.y*blockDim.y + threadIdx.y;	// col index

	if (  m < row_size && n < col_size ) {

		int shuffle_data_index = m * row_stride + n;

		int buffer_data_index = shuffle_index[m] * row_stride + n;

		nn_out_gpu[shuffle_data_index] = train_data_buffer[buffer_data_index];
	}

}

extern "C" void	lbxPrefetchDataAccordingToShuffleIndexGPU( int batch_size, int col_size, int stride, 
	int *shuffle_index, float *train_data_buffer, float *nn_out_gpu ) {
	
	dim3 dimBlock(16, 16);
	dim3 dimGrid( (batch_size + 15)/16, ( col_size + 15)/16 );
	
	PrefetchDataAccordingToShuffleIndexKernel<<< dimGrid, dimBlock >>> ( batch_size, col_size, stride, 
		shuffle_index, train_data_buffer, nn_out_gpu );
	
	gpuErrCheck( cudaPeekAtLastError() );

}

// ----------------------------------- //
// NN Forward & BP Kernels
// ----------------------------------- //

__global__ static void setConstKernel( int N, float *data, float val ) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while(tid < N) {  
        data[tid] = val;
        tid += blockDim.x * gridDim.x;  
    } 

}

extern "C" void	lbxSetConstGPU( int N, float *data, float val ) {
	
	int blockSize = CUDA_THREADS_PER_BLOCK;
 	int gridSize = ( N + blockSize - 1 )/blockSize;
 
	setConstKernel<<< gridSize, blockSize >>> ( N, data, val );

	gpuErrCheck( cudaPeekAtLastError() );
	
}

__global__ static void SigmoidKernel( int N, float *data_in, float *data_out ) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while(tid < N) {  
        data_out[tid] = 1/ (1.0f + expf( -data_in[tid] ) );
        tid += blockDim.x * gridDim.x;  
    } 

}

extern "C" void	lbxSigmoidGPU( int N, float *data_in, float *data_out ) {
	
	int blockSize = CUDA_THREADS_PER_BLOCK;
 	int gridSize = ( N + blockSize - 1 )/blockSize;
 
	SigmoidKernel<<< gridSize, blockSize >>> ( N, data_in, data_out );

	
}

__global__ static void SoftmaxKernel( int vec_dim, int batch_size, int stride, 
	float *data_in, float *data_out ) {
	
	// Refer to kaldi --> cu-kernels.cu --> softmax_reduce
	int j = blockIdx.x;
	// only threads in the same block can communicate!
	int THREADS = blockDim.x;
	// blockDim = threads Num in a block = data access step for a thread
	if (j >= batch_size) return;
	// each block processes a row of data (max, exp, sum and normalize) with threads
	// j -> index of rows for softmax

	// We open CUDA_THREADS_PER_BLOCK threads in default
	__shared__ float aux[CUDA_THREADS_PER_BLOCK];
	// Divide a col into steps of threads
	int steps = (vec_dim + THREADS - 1) / THREADS;

	//copy input to aux
	aux[threadIdx.x] = data_in[threadIdx.x + j*stride];
	for(int i=1; i<steps; ++i) {
		if( (threadIdx.x + i*THREADS < vec_dim) && (aux[threadIdx.x] < data_in[threadIdx.x + i*THREADS + j*stride]) )
			// find max data_in[] within BLOCKs
			aux[threadIdx.x] = data_in[threadIdx.x + i*THREADS + j*stride];
	}

	//get the maximum value
	int nTotalThreads = THREADS;
	__syncthreads();
	
	while(nTotalThreads > 1) {
		int half_point = ( (1 + nTotalThreads) >> 1 );	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < half_point)  {
			// Get the shared value stored by another thread
			if( (threadIdx.x + half_point < nTotalThreads) && (aux[threadIdx.x] < aux[threadIdx.x+half_point]) )
				aux[threadIdx.x] = aux[threadIdx.x + half_point];
		}
		__syncthreads();
		nTotalThreads = ( (1 + nTotalThreads) >> 1 );   // divide by two.
	}
	float max = aux[0];
	__syncthreads();
  
	// subtract max, apply exp, sum up...
	data_out[threadIdx.x + j*stride] = expf( data_in[threadIdx.x + j*stride] - max );
	aux[threadIdx.x] = data_out[threadIdx.x + j*stride];
	for(int i=1; i<steps; i++) {
		if( (threadIdx.x + i*THREADS) < vec_dim) {
		    data_out[threadIdx.x + i*THREADS + j*stride] = expf( data_in[threadIdx.x + i*THREADS + j*stride] - max );
		  	aux[threadIdx.x] += data_out[threadIdx.x + i*THREADS + j*stride];
		}
	}
	
	nTotalThreads = THREADS;
	__syncthreads();
	
	while(nTotalThreads > 1) {
		int half_point = ( (1 + nTotalThreads) >> 1 );   // divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < half_point)  {
			// Get the shared value stored by another thread
			if( (threadIdx.x + half_point) < nTotalThreads)
				aux[threadIdx.x] += aux[threadIdx.x + half_point];
		}
		__syncthreads();
		nTotalThreads = ( (1 + nTotalThreads) >> 1 );   // divide by two.
	}
	  
	float sum = aux[0];
	__syncthreads();

	//normalize by sum...
	for(int i=0; i<steps; i++) {
	  	if( (threadIdx.x + i*THREADS) < vec_dim ) {
		    data_out[threadIdx.x + i*THREADS + j*stride] = data_out[threadIdx.x + i*THREADS + j*stride] / sum;
		}
	}
	
}

extern "C" void	lbxSoftmaxGPU( int vec_dim, int batch_size, int stride, 
	float *data_in, float *data_out ) {
	

	// Each block process a row of data for softmax with min(CUDA_THREADS_PER_BLOCK, vec_dim) threads 
	int ThreadsPerBlock = vec_dim > CUDA_THREADS_PER_BLOCK ? CUDA_THREADS_PER_BLOCK : vec_dim;
	int BlockNum = batch_size;
	
	SoftmaxKernel<<< BlockNum, ThreadsPerBlock >>> ( vec_dim, batch_size, stride, data_in, data_out );

	gpuErrCheck( cudaPeekAtLastError() );
	
}

__global__ static void DiffSigmoidKernel( int N, float *delta_in, float *result_in, float *delta_out ) {
	
	int tid = blockIdx.x*blockDim.x+threadIdx.x;

	while(tid < N) {  
		// y = (1-res)*res*dx
        delta_out[tid] = (1.0f - result_in[tid]) * result_in[tid] * delta_in[tid] ;
        tid += blockDim.x * gridDim.x;  
    } 

}

extern "C" void	lbxDiffSigmoidGPU( int N, float *delta_in, float *result_in, float *delta_out ) {
	
	int blockSize = CUDA_THREADS_PER_BLOCK;
 	int gridSize = ( N + blockSize - 1 )/blockSize;
 
	DiffSigmoidKernel<<< gridSize, blockSize >>> ( N, delta_in, result_in, delta_out );

	gpuErrCheck( cudaPeekAtLastError() );
	
}



__global__ static void NNDiffKernel( int vec_dim, int batch_size, int stride, 
	float *nn_out_gpu, int *tgt_out_index_gpu, float *diff_gpu ) {
	
	int m = blockIdx.x*blockDim.x + threadIdx.x;	// row index
	int n = blockIdx.y*blockDim.y + threadIdx.y;	// col index

	int tid = m * stride + n;

	if (  m < batch_size && n < vec_dim ) 
		diff_gpu[tid] = nn_out_gpu[tid] - (tgt_out_index_gpu[m] == n);

}

extern "C" void	lbxNNDiffGPU( int vec_dim, int batch_size, int stride, float *nn_out_gpu, int *tgt_out_index_gpu, float *diff_gpu ) {
	
	dim3 dimBlock(16, 16);
	dim3 dimGrid( (batch_size + 15)/16, (vec_dim + 15)/16 );

	NNDiffKernel<<< dimGrid, dimBlock >>> ( vec_dim, batch_size, stride, nn_out_gpu, tgt_out_index_gpu, diff_gpu );

	gpuErrCheck( cudaPeekAtLastError() );
	
}

__global__ static void AccuracyEvalKernel( int vec_dim, int batch_size, int stride, 
	float *nn_out, int *tgt_out_index, int *accuracy_gpu ) {

	int j = blockIdx.x;
	// only threads in the same block can communicate!
	int THREADS = blockDim.x;
	// blockDim = threads Num in a block = data access step for a thread
	if (j >= batch_size) return;
	// each block processes a row of data (max, exp, sum and normalize) with threads
	// j -> index of rows for softmax

	// We open CUDA_THREADS_PER_BLOCK threads in default
	__shared__ float aux[CUDA_THREADS_PER_BLOCK];
	// Divide a col into steps of threads
	int steps = (vec_dim + THREADS - 1) / THREADS;

	float tgt_max = nn_out[ tgt_out_index[j] + j*stride ];

	//copy input to aux
	aux[threadIdx.x] = nn_out[threadIdx.x + j*stride];
	for(int i=1; i<steps; ++i) {
		if( (threadIdx.x + i*THREADS < vec_dim) && (aux[threadIdx.x] < nn_out[threadIdx.x + i*THREADS + j*stride]) )
			// find max nn_out[] within BLOCKs
			aux[threadIdx.x] = nn_out[threadIdx.x + i*THREADS + j*stride];
	}

	//get the maximum value
	int nTotalThreads = THREADS;
	__syncthreads();
	
	while(nTotalThreads > 1) {
		int half_point = ( (1 + nTotalThreads) >> 1 );	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < half_point)  {
			// Get the shared value stored by another thread
			if( (threadIdx.x + half_point < nTotalThreads) && (aux[threadIdx.x] < aux[threadIdx.x + half_point]) )
				aux[threadIdx.x] = aux[threadIdx.x + half_point];
		}
		__syncthreads();
		nTotalThreads = ( (1 + nTotalThreads) >> 1 );   // divide by two.
	}
	
	// Only thread 0 execute atomicAdd()
	// In other words, the accuracy++ executes only once in a row (block)
	if( threadIdx.x == 0 && aux[0] == tgt_max ) 
		atomicAdd( accuracy_gpu, 1);

	__syncthreads();

}

extern "C" void	lbxAccuracyEvalGPU( int vec_dim, int batch_size, int stride, 
	float *nn_out_gpu, int *tgt_out_index_gpu, int *accuracy_gpu ) {
	
	int ThreadsPerBlock = vec_dim > CUDA_THREADS_PER_BLOCK ? CUDA_THREADS_PER_BLOCK : vec_dim;
	int BlockNum = batch_size;

	gpuErrCheck( cudaMemset( accuracy_gpu, 0, sizeof(int) ) );

	AccuracyEvalKernel <<< BlockNum, ThreadsPerBlock >>> ( vec_dim, batch_size, stride, 
		nn_out_gpu, tgt_out_index_gpu, accuracy_gpu );

	gpuErrCheck( cudaPeekAtLastError() );
	
}

__global__ static void XentEvalKernel( int batch_size, int stride, 
	float *nn_out, int *tgt_out_index, float *xent_gpu ) {
	
	__shared__ float aux[CUDA_THREADS_PER_BLOCK];

	int THREADS = blockDim.x;
	int steps = (batch_size + THREADS - 1) / THREADS;
	int index;

	//copy data to aux (shared memory)
	aux[threadIdx.x] = -logf( nn_out[ tgt_out_index[threadIdx.x] + threadIdx.x * stride ] + 1e-20);
	for(int i=1; i<steps; ++i) {
		index = threadIdx.x + i*THREADS;
		if( index < batch_size )
			// accumulate to aux
			aux[threadIdx.x] -= logf ( nn_out[ tgt_out_index[index] + index * stride  ] + 1e-20);
	}

	//sum up
	int nTotalThreads = THREADS;
	__syncthreads();
	
	while(nTotalThreads > 1) {
		int half_point = ( (1 + nTotalThreads) >> 1 );	// divide by two
		// only the first half of the threads will be active.
		if (threadIdx.x < half_point && threadIdx.x + half_point < nTotalThreads )  {
			aux[threadIdx.x] += aux[threadIdx.x + half_point];
		}
		__syncthreads();
		nTotalThreads = ( (1 + nTotalThreads) >> 1 );   // divide by two.
	}
	
	if( threadIdx.x == 0 ) 
		*xent_gpu = aux[0];

}

extern "C" void	lbxXentEvalGPU( int batch_size, int stride, 
	float *nn_out_gpu, int *tgt_out_index_gpu, float *xent_gpu ) {
	
	int ThreadsPerBlock = batch_size > CUDA_THREADS_PER_BLOCK ? CUDA_THREADS_PER_BLOCK : batch_size;
	int BlockNum = 1;

	XentEvalKernel <<< BlockNum, ThreadsPerBlock >>> ( batch_size, stride, nn_out_gpu, tgt_out_index_gpu, xent_gpu );
	gpuErrCheck( cudaDeviceSynchronize() );

	gpuErrCheck( cudaPeekAtLastError() );
	
}

// ----------------------------------- //
// Other kernels
// ----------------------------------- //

__global__ static void adaGradKernel( int nn_para_num, float *nn_para_gpu, float *nn_grad_gpu, float *adagrad_gpu, float learn_rate ) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < nn_para_num) {
		adagrad_gpu[tid] += powf( nn_grad_gpu[tid], 2);
    	nn_para_gpu[tid] -= learn_rate * nn_grad_gpu[tid] * rsqrtf(adagrad_gpu[tid]);
		
		tid += blockDim.x * gridDim.x;  
    }
}


extern "C" void lbxAdaGradGPU( int nn_para_num, float *nn_para_gpu, float *nn_grad_gpu, float *adagrad_gpu, float learn_rate ) {

	int blockSize = CUDA_THREADS_PER_BLOCK;
 	int gridSize = ( nn_para_num + blockSize - 1 )/blockSize;
 
	adaGradKernel<<< gridSize, blockSize >>> ( nn_para_num, nn_para_gpu, nn_grad_gpu, adagrad_gpu, learn_rate );

	gpuErrCheck( cudaPeekAtLastError() );

}

__global__ static void adaDeltaKernel( int nn_para_num, float *nn_para_gpu, float *nn_grad_gpu, 
	float *Exp_g2, float *Exp_dx2, float rou, float eps ) {
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	while (tid < nn_para_num) {
	
		float nn_update;
		Exp_g2[tid] = (rou*Exp_g2[tid]) + ( (1-rou)*powf(nn_grad_gpu[tid], 2) );
		nn_update = -sqrtf( Exp_dx2[tid] + eps ) * rsqrtf( Exp_g2[tid] + eps ) * nn_grad_gpu[tid];
		Exp_dx2[tid] = (rou*Exp_dx2[tid]) + ( (1-rou)*powf(nn_update, 2) );
		nn_para_gpu[tid] += nn_update;
		
		tid += blockDim.x * gridDim.x;  
    }
}

extern "C" void lbxAdaDeltaGPU( int nn_para_num, float *nn_para_gpu, float *nn_grad_gpu, 
	float *Exp_g2, float *Exp_dx2, float rou, float eps ) {

	int blockSize = CUDA_THREADS_PER_BLOCK;
 	int gridSize = ( nn_para_num + blockSize - 1 )/blockSize;
 
	adaDeltaKernel<<< gridSize, blockSize >>> ( nn_para_num, nn_para_gpu, nn_grad_gpu, Exp_g2, Exp_dx2, rou, eps );

	gpuErrCheck( cudaPeekAtLastError() );

}
