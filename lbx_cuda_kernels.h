#ifndef LBX_CUDA_KERNEL_H_
#define LBX_CUDA_KERNEL_H_

extern "C" void	lbxSpliceFramesToCreatFeatureGPU( int frame_length, int splice_length, 
	int row_size, int row_stride,
	int *splice_index_gpu,
	float *data_in, float *data_out,
	int data_in_index_bias, int data_in_length );

extern "C" void	lbxAddRowToMatrixThenRescaleGPU( int row_size, int col_size, int row_stride, 
	float *add_row_data, float *rescale_data, float *mat_data );

extern "C" void	lbxPrefetchDataAccordingToShuffleIndexGPU( int batch_size, int col_size, int stride, 
	int *shuffle_index, float *train_data_buffer, float *nn_out_gpu );

extern "C" void	lbxSetConstGPU( int N, float *data, float val );

extern "C" void	lbxSigmoidGPU( int N, float *data_in, float *data_out );

extern "C" void	lbxSoftmaxGPU( int vec_dim, int batch_size, int stride, 
	float *data_in, float *data_out );

extern "C" void	lbxDiffSigmoidGPU( int N, float *delta_in, float *result_in, float *delta_out );

extern "C" void	lbxNNDiffGPU( int vec_dim, int batch_size, int stride, 
	float *nn_out_gpu, int *tgt_out_index_gpu, float *diff_gpu );

extern "C" void	lbxAccuracyEvalGPU( int vec_dim, int batch_size, int stride, 
	float *nn_out_gpu, int *tgt_out_index_gpu, int *accuracy_gpu );

extern "C" void	lbxXentEvalGPU( int batch_size, int stride, 
	float *nn_out_gpu, int *tgt_out_index_gpu, float *xent_gpu );

extern "C" void lbxAdaDeltaGPU( int nn_para_num, float *nn_para_gpu, float *nn_grad_gpu, 
	float *Exp_g2, float *Exp_dx2, float rou, float eps );
	
#endif