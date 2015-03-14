#ifndef MAX_STRING_SIZE
#define MAX_STRING_SIZE 256
#endif 

#ifndef IO_READ_BUF_SIZE
#define IO_READ_BUF_SIZE 1024
#endif 

#ifndef LBX_IO_H_
#define LBX_IO_H_

#include "cnnFunc.h"

extern FILE *file_pointer_key;
extern FILE *file_pointer_label;
extern FILE *file_pointer_value;

extern int size_row_left_to_read_this_utter;
extern bool finished_reading_a_utter;

void ShuffleArray_Fisher_Yates(int* arr, int len);

// the pre-processed 924-dim training data will be stored in train_data_buffer_gpu;
// temp_train_data_gpu buffer the original 84-dim data for each IO operation
int lbxReadTrainData( float *train_data_buffer_gpu, float *temp_train_data_gpu,
    int *label_cpu, 
    int frame_length, int splice_length, 
    int pre_frame_MAX, int post_frame_MAX,
    int pre_frame_LENGTH, int post_frame_LENGTH,
    int *splice_index_gpu, 
    int buffer_size, int buffer_row_stride,
    int *size_row_filled_in_buffer,
    long long *utter_count, int *utter_progress );

int ReadKaldiCNNnet( const char *Nnet_FILE_name,
	float **nn_para_cpu_func_output, float **cnn_para_cpu_output,
	int **nn_in_dim_func_output, int **nn_out_dim_func_output,
	volatile cnn_params *params);

int SaveKaldiCNNnet( const char *Nnet_FILE_name,
	int nn_layer_num, int nn_para_num, 
	float *nn_para_cpu,
	int *nn_in_dim, int *nn_out_dim,
	float *cnn_para_cpu, cnn_params *params);

int lbxReadFeatureTransform( const char *Nnet_FILE_name,
    int *feature_in_dim, int *feature_out_dim,
    int *splice_length,
    int **splice_index,
    float **add_shift_data, 
    float **rescale_data );

#endif
