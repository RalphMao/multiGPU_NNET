#ifndef progress_step
#define progress_step 3600*100
#endif

#ifndef LBX_NN_H_
#define LBX_NN_H_

void lbxNetEval( float *nn_out_gpu, int *tgt_out_index_gpu, 
	float *diff_gpu, 
    int nn_out_dim, int batch_size, 
    int nn_out_stride,
    int *correct_gpu, float *xent_gpu,
    float *loss_all,
    long long *correct_all, 
    long long * frames_all,
    int *frames_progress,
    float *loss_progress );

void lbxNetPropagation( float *nn_para_gpu, float *nn_hidden_result_gpu, 
	float *train_data, float *nn_out_gpu,   
	int layer_num, int batch_size, 
	int *nn_in_dim, int *nn_out_dim, 
	int train_data_stride, int nn_out_stride, 
	int *weight_stride, int *bias_stride, int *result_stride, 
	float *vec_one_gpu );

void lbxNetBackProp( float *nn_para_gpu, 
	float *nn_hidden_result_gpu, float *nn_delta_gpu,
	float *train_data, float *obj_diff,   
	int layer_num, int batch_size, 
	int *nn_in_dim, int *nn_out_dim, 
	int train_data_stride, int obj_diff_stride, 
	int *weight_stride, int *bias_stride, int *result_stride, 
	float *mid_delta_gpu, int mid_delta_stride, //new
	float learn_rate, 
	float *vec_one_gpu );

#endif
