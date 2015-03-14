#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "gpuErrCheck_INLINE_H_.h"
#include "lbx_cuda_kernels.h"
#include "cnn_io.h"


// Read the token and data of type "int"
int read_int(int *data_pointer, const char *str_cmp, FILE *Nnet_file_pointer)
{
    char read_temp[MAX_STRING_SIZE];
    fscanf( Nnet_file_pointer, "%s", read_temp);
    if (strcasecmp(read_temp, str_cmp) == 0) { 
	char yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(data_pointer, sizeof(int), 1, Nnet_file_pointer);
    }
    else {
	printf("Expect Token:%s, Get Token:%s", str_cmp, read_temp);
	return 1 ;
    }
    return 0 ;
}

int write_data(void *data_pointer, int word_length, int length, FILE *Nnet_file_pointer)
{
    fputc(4, Nnet_file_pointer);
	int num_write_this_time = fwrite( data_pointer, 
		word_length, length,
		Nnet_file_pointer );

	if ( num_write_this_time != length ) {
	    printf("Write Failed\n");
	    return -2;
	}
	return 0;
}
// throw away useless data or do nothing
int throw_element(char *str_cmp, FILE *Nnet_file_pointer)
{
    char read_temp[MAX_STRING_SIZE];
    long int file_location = ftell(Nnet_file_pointer);
    int trash_space;

    fscanf( Nnet_file_pointer, "%s", read_temp);
    if (strcasecmp(read_temp, str_cmp) == 0) {
	char yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&trash_space, sizeof(int), 1, Nnet_file_pointer);
    }
    else { //return to the original position
	fseek(Nnet_file_pointer, file_location, SEEK_SET); 
    }
    return 0;
}

void ShuffleArray_Fisher_Yates(int* arr, int len) {

    int i = len-1, j;
    int temp;

    srand( (unsigned) time(NULL) );

    if ( i == 0 ) return;
    while ( i > 0 ) {
	j = rand() % i;
	temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
	i--;
    }
}

int lbxReadTrainData( float *train_data_buffer_gpu, float *temp_train_data_gpu,
	int *label_cpu, 
	int frame_length, int splice_length, 
	int pre_frame_MAX, int post_frame_MAX,
	int pre_frame_LENGTH, int post_frame_LENGTH,
	int *splice_index_gpu,
	int buffer_size, int buffer_row_stride,
	int *size_row_filled_in_buffer,
	long long *utter_count, int *utter_progress )

{

    float *read_data_buffer = (float *)malloc( IO_READ_BUF_SIZE * frame_length * sizeof(float) );

    char read_temp[MAX_STRING_SIZE];
    char key_label[MAX_STRING_SIZE];
    char *char_pointer_to_a_string;

    long long value_index_in_the_file;

    int row_size, col_size;
    int num_read_this_time;
    int read_size;

    // temp value for loop
    int label_read_count;

    int size_row_read_this_time;
    int size_row_to_be_filled;

    if( feof( file_pointer_key ) ) {
	printf("End of the Key FILE of Training Data!\n");
	fflush(stdout);
	return -1;
    }

    if( !finished_reading_a_utter && file_pointer_value == NULL ) {
	printf("Can't Read Value File!\n");
	fflush(stdout);
	return -2;
    }

    /*
       int active_GPU_ID;
       cudaGetDevice( &active_GPU_ID );
       printf("GPU[%d] Begin Reading Data\n", active_GPU_ID );
       */

    while ( (*size_row_filled_in_buffer) < buffer_size ) {

	if( finished_reading_a_utter ) {

	    if( fscanf( file_pointer_key, "%s%s", key_label, read_temp) == EOF ) {
		printf("End of the Key FILE of Training Data!\n");
		fflush(stdout);
		return -1;
	    }
	    // printf("key: %s, value: %s\n", key_label, read_temp);

	    // find the value index in the key
	    char_pointer_to_a_string = strstr( read_temp, ":" );

	    *char_pointer_to_a_string = '\0';

	    // printf("lbx_dir = %s\n", read_temp);
	    file_pointer_value = fopen( read_temp, "r" );

	    if( file_pointer_value == NULL ) {
		printf("VALUE FILE OPEN ERROR!\n");
		fflush(stdout);
		return -2;
	    }
	    // printf("Value File is Open!\n");

	    // atoi may overflow!!! we use long long and atoll here 
	    value_index_in_the_file = atoll( char_pointer_to_a_string + 1 );
	    // printf("%lld\n", value_index_in_the_file);

	    if( value_index_in_the_file < 0 ) {
		printf("Illegal value index\n");
		fflush(stdout);
		return -3;
	    }

	    fscanf(file_pointer_label, "%s", read_temp);
	    while( strcmp( read_temp, key_label ) < 0 ) {
		printf("Label = %s, Label_Index < Feats_Index, Looking for Label Data...\n", read_temp);
		do {
		    read_size = fgetc( file_pointer_label );
		} while ( read_size != '\n' );
		fscanf(file_pointer_label, "%s", read_temp);    
	    }

	    // printf("label_index = %s\n", read_temp);
	    if( strcmp( read_temp, key_label ) != 0 ) {
		printf( "Label: %s, Feat: %s\n", read_temp, key_label );
		printf( "Mismatch of Label and Feat\n" );
		fflush(stdout);
		return -4;
	    }

	    // begin to read the value of training data
	    fseek( file_pointer_value, value_index_in_the_file + 1, SEEK_SET );

	    // 'BFM'
	    fgets( read_temp, 4, file_pointer_value );
	    if ( strcmp( read_temp, "BFM" ) != 0 ) {
		printf("Type of the Training Data is NOT FLOAT!\n");
		fflush(stdout);
		return -4;
	    }

	    // read the blank space
	    fgetc( file_pointer_value );

	    // first '4' for row size
	    read_size =  fgetc( file_pointer_value );
	    if( read_size != 4 ) {
		printf("Type of the Row Size is NOT INT!\n");
		fflush(stdout);
		return -5;
	    }

	    num_read_this_time = fread( &row_size, sizeof(int), 1, file_pointer_value );
	    if( row_size < 0 ) {
		printf("Row = %d, Illegal Size!\n", row_size);
		fflush(stdout);
		return -6;
	    }
	    // printf("Num Read = %d, Row Num = %d\n", num_read_this_time, row_size );

	    // second '4' for col size
	    read_size =  fgetc( file_pointer_value );
	    if( read_size != 4 ) {
		printf("Type of the Col Size is NOT INT!\n");
		fflush(stdout);
		return -5;
	    }

	    num_read_this_time = fread( &col_size, sizeof(int), 1, file_pointer_value );
	    if( col_size != frame_length ) {
		printf("Col = %d, Error Size!\n", col_size);
		fflush(stdout);
		return -6;
	    }
	    // printf("Num Read = %d, Col Num = %d\n", num_read_this_time, col_size );

	    finished_reading_a_utter = false;

	    size_row_left_to_read_this_utter = row_size;

	    size_row_read_this_time = size_row_left_to_read_this_utter > IO_READ_BUF_SIZE ? IO_READ_BUF_SIZE : size_row_left_to_read_this_utter;

	    // read the first block of frames
	    num_read_this_time = fread( read_data_buffer, sizeof(float), size_row_read_this_time * frame_length, file_pointer_value );
	    // printf("Read %d Features\n", num_read_this_time );

	    if( num_read_this_time != (size_row_read_this_time * frame_length) ) {
		printf("Read Training Data Error!\n");
		fflush(stdout);
		return -7;
	    }

	    // offset (PRE+POST FRAME) rows
	    gpuErrCheck( cudaMemcpy( temp_train_data_gpu + ( (pre_frame_MAX + post_frame_MAX) * frame_length ), 
			read_data_buffer, 
			size_row_read_this_time * frame_length * sizeof(float),
			cudaMemcpyHostToDevice ) );

	    // waiting for post_frame_LENGTH
	    size_row_to_be_filled = size_row_read_this_time == row_size ? row_size : (size_row_read_this_time - post_frame_LENGTH);
	    // printf( "size_row_to_be_filled = %d\n", size_row_to_be_filled );

	    lbxSpliceFramesToCreatFeatureGPU( frame_length, splice_length,
		    size_row_to_be_filled, buffer_row_stride, 
		    splice_index_gpu,
		    temp_train_data_gpu + ( (pre_frame_MAX + post_frame_MAX) * frame_length), 
		    train_data_buffer_gpu + ( (*size_row_filled_in_buffer) * buffer_row_stride), 
		    0, row_size );

	    // read the corresponding label data
	    for( label_read_count = 0; label_read_count< size_row_to_be_filled; label_read_count++ ) {
		fscanf(file_pointer_label, "%d", label_cpu + (*size_row_filled_in_buffer) + label_read_count );
	    }

	    size_row_left_to_read_this_utter -= size_row_read_this_time;
	    (*size_row_filled_in_buffer) += size_row_to_be_filled;
	    // printf("left %d rows in this utter, fill %d in GPU\n", size_row_left_to_read_this_utter, *size_row_filled_in_buffer);

	}

	while( size_row_left_to_read_this_utter > 0 && (*size_row_filled_in_buffer) < buffer_size ) {

	    // copy the last (PRE+POST FRAMES) rows to the top of the buffer
	    gpuErrCheck( cudaMemcpy( temp_train_data_gpu, 
			temp_train_data_gpu + (IO_READ_BUF_SIZE * frame_length),
			(pre_frame_MAX + post_frame_MAX) * frame_length * sizeof(float),
			cudaMemcpyDeviceToDevice ) );

	    size_row_read_this_time = size_row_left_to_read_this_utter > IO_READ_BUF_SIZE ? IO_READ_BUF_SIZE : size_row_left_to_read_this_utter;

	    // printf("left_utter = %d, read_this_time = %d\n", size_row_left_to_read_this_utter, size_row_read_this_time);

	    // read training data
	    num_read_this_time = fread( read_data_buffer, sizeof(float), size_row_read_this_time * frame_length, file_pointer_value );
	    // printf("Read %d Features\n", num_read_this_time );

	    if( num_read_this_time != (size_row_read_this_time * frame_length) ) {
		printf("Read Training Data Error!\n");
		fflush(stdout);
		return -7;
	    }

	    // offset 10 rows
	    gpuErrCheck( cudaMemcpy( temp_train_data_gpu + ( (pre_frame_MAX + post_frame_MAX) * frame_length ), 
			read_data_buffer, 
			size_row_read_this_time * frame_length * sizeof(float),
			cudaMemcpyHostToDevice ) );

	    size_row_to_be_filled = size_row_read_this_time == size_row_left_to_read_this_utter ? 
		( size_row_left_to_read_this_utter + post_frame_LENGTH ) : size_row_read_this_time;
	    // printf( "size_row_to_be_filled = %d\n", size_row_to_be_filled );

	    lbxSpliceFramesToCreatFeatureGPU( frame_length, splice_length,
		    size_row_to_be_filled, buffer_row_stride,
		    splice_index_gpu,
		    temp_train_data_gpu,
		    train_data_buffer_gpu + ( (*size_row_filled_in_buffer) * buffer_row_stride ), 
		    pre_frame_MAX, size_row_left_to_read_this_utter + (pre_frame_LENGTH + post_frame_LENGTH) );

	    // read the corresponding label data
	    for( label_read_count = 0; label_read_count< size_row_to_be_filled; label_read_count++ ) {
		fscanf(file_pointer_label, "%d", label_cpu + (*size_row_filled_in_buffer) + label_read_count );
	    }

	    size_row_left_to_read_this_utter -= size_row_read_this_time;
	    (*size_row_filled_in_buffer) += size_row_to_be_filled;
	    // printf("left %d rows in this utter, fill %d rows in GPU\n", size_row_left_to_read_this_utter, *size_row_filled_in_buffer);

	}

	if( size_row_left_to_read_this_utter == 0 ) {

	    // printf("Finish Reading a Utter\n");
	    finished_reading_a_utter = true;

	    (*utter_count)++;
	    (*utter_progress)++;

	    fclose(file_pointer_value);

	}

    }

    // printf("IO Function Finish this Time\n");

    free(read_data_buffer);

    return 0;

}

int ReadKaldiCNNnet( const char *Nnet_FILE_name,
	float **nn_para_cpu_func_output, float **cnn_para_cpu_output,
	int **nn_in_dim_func_output, int **nn_out_dim_func_output,
	volatile cnn_params *params)
{

    float *nn_para_cpu;
    int *nn_in_dim;
    int *nn_out_dim;

    float *nn_para_cpu_BAK_for_dynamic_malloc;
    int *nn_in_dim_BAK_for_dynamic_malloc;
    int *nn_out_dim_BAK_for_dynamic_malloc;

    int data_temp;

    int nn_layer_num = 0;
    int nn_para_num_current = 0;

    int nn_layer_count;

    FILE *Nnet_file_pointer = fopen( Nnet_FILE_name, "rb" );
    if ( Nnet_file_pointer ==NULL) {
	printf("Can't OPEN Nnet FILE!\n");
	return -1;
    }

    char read_temp[MAX_STRING_SIZE] = "Nothing";

    int num_read_this_time;
    int read_char;

    int nn_in_dim_temp, nn_out_dim_temp;
    int row_size, col_size;
    int vec_dim;

    char yyread_char; 

    // read '\0'
    read_char = fgetc( Nnet_file_pointer );

    read_char = fgetc( Nnet_file_pointer );
    if( read_char != 'B' ) {
	printf("Not Binary FILE!\n" );
	return -2;
    }
    // printf("Binary FILE\n");

    // read “<Nnet>”
    fscanf( Nnet_file_pointer, "%s", read_temp);
    if( strcmp( read_temp, "<Nnet>" ) ) {
	printf("Begining of the File: %s\n", read_temp );
	printf("Illegal Net Data\n");
	return -2;
    }
    // printf("Begining of the File: %s\n", read_temp );

    printf( "-------------------------------------\n" );
    printf( "      Start Reading Kaldi Nnet\n" );
    printf( "-------------------------------------\n" );

    // read the space
    read_char = fgetc( Nnet_file_pointer );

    // read the convolutional parts
    fscanf( Nnet_file_pointer, "%s", read_temp);

    if (strcasecmp(read_temp, "<convolutionalcomponent>")) {
	printf("Layer Type = %s\n", read_temp);
	if (strcasecmp(read_temp, "<affinetransform>")==0) {
	    printf("This program does not support DNN\n");
	}
	else {
	    printf("Can't support this file. Please Contact the Designers....\n");
	}
	return -2;
    } else {
	params->layer_num = 1;
	int n = 1;

    params->patch_stride  = (int*)malloc(sizeof(int)*n);
    params->patch_dim     = (int*)malloc(sizeof(int)*n);
    params->patch_step    = (int*)malloc(sizeof(int)*n);
    params->filter_section  = (int*)malloc(sizeof(int)*n); 
    params->filter_step      = (int*)malloc(sizeof(int)*n); 
    params->pool_size       = (int*)malloc(sizeof(int)*n); 
    params->pool_step      = (int*)malloc(sizeof(int)*n);
    params->pool_stride    = (int*)malloc(sizeof(int)*n);
    params->cnn_in_dim    = (int*)malloc(sizeof(int)*n); 
    params->cnn_out_dim  = (int*)malloc(sizeof(int)*n); 
    params->pool_out_dim = (int*)malloc(sizeof(int)*n); 

	yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(params->cnn_out_dim, sizeof(int), 1, Nnet_file_pointer);

	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(params->cnn_in_dim, sizeof(int), 1, Nnet_file_pointer);

	if (read_int(params->patch_dim, "<patchdim>", Nnet_file_pointer)) return -2;
	if (read_int(params->patch_step, "<patchstep>", Nnet_file_pointer)) return -2;
	if (read_int(params->patch_stride, "<patchstride>", Nnet_file_pointer)) return -2;
	if (read_int(params->filter_section, "<filtersection>", Nnet_file_pointer)) return -2;
	if (read_int(params->filter_step, "<sectionstep>", Nnet_file_pointer)) return -2;

	if (throw_element("<learnratecoef>", Nnet_file_pointer)) return -2;
	if (throw_element("<biaslearnratecoef>", Nnet_file_pointer)) return -2;
	if (throw_element("<maxnorm>", Nnet_file_pointer)) return -2;

    int num_splice = params->cnn_in_dim[0] / params->patch_stride[0];
    int num_sections = 1 + (params->patch_stride[0] - params->filter_section[0])/ params->filter_step[0];
    int num_patches = 1 + (params->filter_section[0] - params->patch_dim[0]) / params->patch_step[0];
    int num_filters = params->cnn_out_dim[0] / num_patches;
    int num_filters_sec = num_filters/num_sections;
    int filter_dim = num_splice * params->patch_dim[0];

    // Sanity Check
    if (params->cnn_in_dim[0] % params->patch_stride[0] != 0)
    {
	printf("CNN in dim does not match with patch stride!\n");
	return -2;
    }

    if ((params->patch_stride[0] - params->patch_dim[0]) % params->patch_step[0] != 0)
    {
	printf("Patch number is not integer!\n");
	return -2;
    }

    if (params->cnn_out_dim[0] % num_patches != 0)
    {
	printf("CNN out dim does not match with patch number!\n");
	return -2;
    }

	// Read Filters
	fscanf(Nnet_file_pointer, "%s", read_temp);
	if (strcasecmp(read_temp, "<filters>")) {
	    printf("Unknow syntax:%s\n", read_temp);
	    return 2;
	}
	yyread_char = fgetc( Nnet_file_pointer );

	fscanf(Nnet_file_pointer, "%s", read_temp);
	if (strcasecmp(read_temp, "FM")) {
	    printf("Unknow syntax:%s\n", read_temp);
	    return 2;
	}
	yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&row_size, sizeof(int), 1, Nnet_file_pointer);
	if( row_size != num_filters) {
	    printf("row_size = %d, num_filters= %d\n", row_size, num_filters);
	    printf("Mismatch!\n");
	    return -4;
	}
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&col_size, sizeof(int), 1, Nnet_file_pointer);
	if( col_size != filter_dim) {
	    printf("col_size = %d, filter_dim= %d\n", row_size, filter_dim);
	    printf("Mismatch!\n");
	    return -4;
	}

	params->cnn_para_num = (filter_dim + 1) * num_filters;
	
	*cnn_para_cpu_output = (float *)malloc(params->cnn_para_num * sizeof(float));
	fread(*cnn_para_cpu_output, sizeof(float), filter_dim * num_filters, Nnet_file_pointer);

	// Read Bias
	fscanf(Nnet_file_pointer, "%s", read_temp);
	if (strcasecmp(read_temp, "<bias>")) {
	    printf("Unknow syntax:%s\n", read_temp);
	    return 2;
	}
	yyread_char = fgetc( Nnet_file_pointer );

	fscanf(Nnet_file_pointer, "%s", read_temp);
	if (strcasecmp(read_temp, "FV")) {
	    printf("Unknow syntax:%s\n", read_temp);
	    return 2;
	}
	yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&row_size, sizeof(int), 1, Nnet_file_pointer);
	if( row_size != num_filters) {
	    printf("row_size = %d, num_filters= %d\n", row_size, num_filters);
	    printf("Mismatch!\n");
	    return -4;
	}

	fread(*cnn_para_cpu_output + filter_dim * num_filters, sizeof(float), num_filters, Nnet_file_pointer);

    }

    // read max pooling part
    fscanf( Nnet_file_pointer, "%s", read_temp);

    if (strcasecmp(read_temp, "<maxpoolingcomponent>")) {
	printf("Layer Type = %s\n", read_temp);
	printf("Can't support this file. Please Contact the Designers....\n");
	return -2;
	}
    else {
	yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(params->pool_out_dim, sizeof(int), 1, Nnet_file_pointer);

	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&data_temp, sizeof(int), 1, Nnet_file_pointer);
	if (data_temp != params->cnn_out_dim[0])
	{
	    printf("Conv out and pool in does not match!\n");
	    return 1;
	}

	if (read_int(params->pool_size, "<poolsize>", Nnet_file_pointer)) return -2;
	if (read_int(params->pool_step, "<poolstep>", Nnet_file_pointer)) return -2;
	if (read_int(params->pool_stride, "<poolstride>", Nnet_file_pointer)) return -2;

	// Sanity Check
	int num_patches_pool = params->cnn_out_dim[0] / params->pool_stride[0];
	int num_pools = 1 + (num_patches_pool - params->pool_size[0] ) / params->pool_step[0];

	if (params->cnn_out_dim[0] % params->pool_stride[0] != 0)
	{
	    printf("Pool in dim does not match with pool stride!\n");
	    return -2;
	}

	if ((num_patches_pool - params->pool_size[0]) % params->pool_step[0] != 0)
	{
	    printf("Pool Stride number is not an integer!\n");
	    return -2;
	}

	if (params->pool_out_dim[0] % params->pool_stride[0] != 0)
	{
	    printf("Pool out dim does not match with pool stride!\n");
	    return -2;
	}

    }

    // read sigmoid layer
    fscanf( Nnet_file_pointer, "%s", read_temp);

    if (strcasecmp(read_temp, "<sigmoid>")) {
	printf("Layer Type = %s\n", read_temp);
	printf("Can't support this file. Please Contact the Designers....\n");
	return -2;
	}
    else {
	yyread_char = fgetc( Nnet_file_pointer );
	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&data_temp, sizeof(int), 1, Nnet_file_pointer);
	if (data_temp != params->pool_out_dim[0])
	{
	    printf("Conv out and pool in does not match!\n");
	    return 1;
	}

	yyread_char = fgetc( Nnet_file_pointer );
	if (yyread_char != 4) {
	    printf("Unknow Format\n");
	    return 2;
	}
	fread(&data_temp, sizeof(int), 1, Nnet_file_pointer);
	if (data_temp != params->pool_out_dim[0])
	{
	    printf("Conv out and pool in does not match!\n");
	    return 1;
	}
    }
	
    // Read DNN parts
    while( !feof( Nnet_file_pointer ) ) {

	// read NN layer type
	fscanf( Nnet_file_pointer, "%s", read_temp);

	if( strcmp( read_temp, "<affinetransform>" ) && strcmp( read_temp, "<AffineTransform>" ) ) {
	    printf("Layer Type = %s\n", read_temp);
	    printf("Can't support this file. Please Contact the Designers....\n");
	    return -2;
	}

	nn_layer_num++;

	printf("Layer %2d: %s + ", nn_layer_num, read_temp );

	// read the space
	yyread_char = fgetc( Nnet_file_pointer );

	// '4'
	read_char = fgetc( Nnet_file_pointer );
	if (read_char != 4) {
	    printf("Unknown Format!\n");
	    return -3;
	}
	// printf("%d\n", read_char );

	// nn_out_dim_temp
	num_read_this_time = fread( &nn_out_dim_temp, sizeof(int), 1, Nnet_file_pointer );
	// printf("nn_out_dim_temp = %d\n", nn_out_dim_temp );

	// '4'
	read_char = fgetc( Nnet_file_pointer );
	if (read_char != 4) {
	    printf("Unknown Format!\n");
	    return -3;
	}
	// printf("%d\n", read_char );

	// nn_in_dim_temp
	num_read_this_time = fread( &nn_in_dim_temp, sizeof(int), 1, Nnet_file_pointer );
	// printf("nn_in_dim_temp = %d\n", nn_in_dim_temp );

	// read 'FM' (Float Matrix)
	fscanf( Nnet_file_pointer, "%s", read_temp );

	if ( !strcmp( read_temp, "<LearnRateCoef>" ) || !strcmp( read_temp, "<learnratecoef>" ) ) {

	    // read the space
	    read_char = fgetc( Nnet_file_pointer );

	    // '4'
	    read_char = fgetc( Nnet_file_pointer );

	    num_read_this_time = fread( &row_size, sizeof(int), 1, Nnet_file_pointer );

	    // read '<BiasLearnRateCoef>'
	    fscanf( Nnet_file_pointer, "%s", read_temp );

	    // read the space
	    read_char = fgetc( Nnet_file_pointer );

	    // '4'
	    read_char = fgetc( Nnet_file_pointer );

	    num_read_this_time = fread( &row_size, sizeof(int), 1, Nnet_file_pointer );

	    fscanf( Nnet_file_pointer, "%s", read_temp );

	}


	if ( strcmp( read_temp, "FM" ) ) {
	    if (!strcmp(read_temp, "<MaxNorm>") || !strcmp(read_temp, "<maxnorm>"))
	    {
		// read the space
		read_char = fgetc( Nnet_file_pointer );

		// '4'
		read_char = fgetc( Nnet_file_pointer );

		num_read_this_time = fread( &row_size, sizeof(int), 1, Nnet_file_pointer );

		fscanf( Nnet_file_pointer, "%s", read_temp );
	    }
	    else
	    {
		printf("|%s|\n",read_temp);
		printf("Data Type = %s, Should be 'FM'...\n", read_temp);
		printf("Can't support this file. Please Contact the Designers....\n");
		return -3;
	    }
	}
	// printf("%s\n", read_temp );

	// read the space
	read_char = fgetc( Nnet_file_pointer );

	// '4'
	read_char = fgetc( Nnet_file_pointer );
	if (read_char != 4) {
	    printf("Unknown Format!\n");
	    return -3;
	}
	// printf("%d\n", read_char );

	// row_size
	num_read_this_time = fread( &row_size, sizeof(int), 1, Nnet_file_pointer );
	if( row_size != nn_out_dim_temp ) {
	    printf("row_size = %d, nn_out_dim_temp = %d\n", row_size, nn_out_dim_temp);
	    printf("Mismatch!\n");
	    return -4;
	}
	// printf("row_size = %d\n", row_size );

	// '4'
	read_char = fgetc( Nnet_file_pointer );
	if (read_char != 4) {
	    printf("Unknown Format!\n");
	    return -3;
	}
	// printf("%d\n", read_char );

	// col_size
	num_read_this_time = fread( &col_size, sizeof(int), 1, Nnet_file_pointer );
	if( col_size != nn_in_dim_temp ) {
	    printf("col_size = %d, nn_in_dim_temp = %d\n", col_size, nn_in_dim_temp);
	    printf("Mismatch!\n");
	    return -4;
	}
	// printf("col_size = %d\n", col_size );

	if( nn_para_num_current !=0 ) {

	    nn_para_cpu_BAK_for_dynamic_malloc = (float *)malloc( nn_para_num_current * sizeof(float) );

	    memcpy( nn_para_cpu_BAK_for_dynamic_malloc, 
		    nn_para_cpu,
		    nn_para_num_current * sizeof(float) );
	    free( nn_para_cpu );

	    nn_para_cpu = (float *)malloc( ( nn_para_num_current + ((col_size +1) * row_size) ) * sizeof(float) );
	    memcpy( nn_para_cpu,
		    nn_para_cpu_BAK_for_dynamic_malloc, 
		    nn_para_num_current * sizeof(float) );

	    free( nn_para_cpu_BAK_for_dynamic_malloc );

	    // modify nn_in_dim & nn_out_dim

	    nn_in_dim_BAK_for_dynamic_malloc  = (int *)malloc( ( nn_layer_num - 1 ) * sizeof(int) );
	    nn_out_dim_BAK_for_dynamic_malloc = (int *)malloc( ( nn_layer_num - 1 )* sizeof(int) );

	    for( nn_layer_count = 0; nn_layer_count < (nn_layer_num-1); nn_layer_count++ ) {
		nn_in_dim_BAK_for_dynamic_malloc[nn_layer_count]  = nn_in_dim[nn_layer_count];
		nn_out_dim_BAK_for_dynamic_malloc[nn_layer_count] = nn_out_dim[nn_layer_count];
	    }

	    free(nn_in_dim);
	    free(nn_out_dim);

	    nn_in_dim = (int *)malloc( nn_layer_num * sizeof(int) );
	    nn_out_dim = (int *)malloc( nn_layer_num * sizeof(int) );

	    for( nn_layer_count = 0; nn_layer_count < (nn_layer_num-1); nn_layer_count++ ) {
		nn_in_dim[nn_layer_count]  = nn_in_dim_BAK_for_dynamic_malloc[nn_layer_count];
		nn_out_dim[nn_layer_count] = nn_out_dim_BAK_for_dynamic_malloc[nn_layer_count];
	    }

	    nn_in_dim[nn_layer_num-1] = col_size;
	    nn_out_dim[nn_layer_num-1] = row_size;

	    free(nn_in_dim_BAK_for_dynamic_malloc);
	    free(nn_out_dim_BAK_for_dynamic_malloc);

	} else {

	    nn_para_cpu = (float *)malloc( ( (col_size +1) * row_size ) * sizeof(float) );

	    nn_in_dim = (int *)malloc( sizeof(int) );
	    nn_out_dim = (int *)malloc( sizeof(int) );

	    nn_in_dim[0] = col_size;
	    nn_out_dim[0] = row_size;

	}

	// read the weight data
	num_read_this_time = fread( nn_para_cpu + nn_para_num_current, sizeof(float), row_size * col_size, Nnet_file_pointer );
	if ( num_read_this_time != (row_size * col_size) ) {
	    printf("Error! Only read %d data of Weights\n", num_read_this_time );
	}

	nn_para_num_current += ( row_size * col_size );
	// printf("Current nn_para_num = %d\n", nn_para_num_current );

	// read 'FV' (Float Vector)
	fscanf( Nnet_file_pointer, "%s", read_temp);
	if( strcmp( read_temp, "FV" ) ) {
	    printf("Data Type = %s, Should be 'FV'...\n", read_temp);
	    printf("Can't support this file. Please Contact the Designers....\n");
	    return -3;
	}
	// printf("%s\n", read_temp );

	// read the space
	read_char = fgetc( Nnet_file_pointer );

	// '4'
	read_char = fgetc( Nnet_file_pointer );
	if (read_char != 4) {
	    printf("Unknown Format!\n");
	    return -3;
	}
	// printf("%d\n", read_char );

	// vec_dim
	num_read_this_time = fread( &vec_dim, sizeof(int), 1, Nnet_file_pointer );
	if( vec_dim != nn_out_dim_temp ) {
	    printf("vec_dim = %d, nn_out_dim_temp = %d\n", vec_dim, nn_out_dim_temp);
	    printf("Mismatch!\n");
	    return -4;
	}
	// printf("bias_size = %d\n", vec_dim );

	// read the bias data
	num_read_this_time = fread( nn_para_cpu + nn_para_num_current, sizeof(float), vec_dim, Nnet_file_pointer );
	if ( num_read_this_time != vec_dim ) {
	    printf("Error! Only read %d data of Bias\n", num_read_this_time );
	}

	nn_para_num_current += row_size;
	// printf("Current nn_para_num = %d\n", nn_para_num_current );

	// read activation function type
	fscanf( Nnet_file_pointer, "%s", read_temp);
	if( !strcmp( read_temp, "<sigmoid>" ) || !strcmp( read_temp, "<softmax>" ) || 
		!strcmp( read_temp, "<Sigmoid>" ) || !strcmp( read_temp, "<Softmax>" ) ) {

	    printf("%s: ", read_temp );
	    printf("Input Dim = %5d, Output Dim = %5d\n", nn_in_dim[nn_layer_num-1], nn_out_dim[nn_layer_num-1] );

	    // read the space
	    read_char = fgetc( Nnet_file_pointer );

	    // '4'
	    read_char = fgetc( Nnet_file_pointer );
	    if (read_char != 4) {
		printf("Unknown Format!\n");
		return -3;
	    }
	    // printf("%d\n", read_char );

	    // nn_out_dim_temp
	    num_read_this_time = fread( &nn_out_dim_temp, sizeof(int), 1, Nnet_file_pointer );
	    if (  nn_out_dim_temp != vec_dim ) {
		printf("Activation Function Output Dim = %d, Mismatch!\n", nn_out_dim_temp );
	    }
	    // printf("nn_out_dim_temp = %d\n", nn_out_dim_temp );

	    // '4'
	    read_char = fgetc( Nnet_file_pointer );
	    if (read_char != 4) {
		printf("Unknown Format!\n");
		return -3;
	    }
	    // printf("%d\n", read_char );

	    // nn_in_dim_temp
	    num_read_this_time = fread( &nn_in_dim_temp, sizeof(int), 1, Nnet_file_pointer );
	    if (  nn_in_dim_temp != vec_dim ) {
		printf("Activation Function Input Dim = %d, Mismatch!\n", nn_in_dim_temp );
	    }
	    // printf("nn_in_dim_temp = %d\n", nn_in_dim_temp );

	    if ( !strcmp( read_temp, "<softmax>" ) || !strcmp( read_temp, "<Softmax>" ) ) {
		break;
	    }

	} else {

	    printf("Layer Type = %s\n", read_temp);
	    printf("Can't support this file. Please Contact the Designers....\n");
	    return -2;
	}

    }

    // read “<\Nnet>”
    fscanf( Nnet_file_pointer, "%s", read_temp);
    if( strcmp( read_temp, "</Nnet>" ) ) {
	printf("End of the File: %s\n", read_temp );
	printf("Illegal Net Data\n");
	return -2;
    }
    // printf("End of the File: %s\n", read_temp );

    // Check the consistency of CNN and DNN 
    if (params->pool_out_dim[0] != nn_in_dim[0]) {
	printf("CNN out dim and DNN in dim do not match!\n");
	return 1;
    }

    printf( "-------------------------------------\n" );
    printf( "     Load Kaldi Nnet Sucessfully\n" );
    printf( "-------------------------------------\n" );
    // printf( "nn_para_num = %d\n", nn_para_num_current );
    // printf( "-------------------------------------\n" );

    fclose( Nnet_file_pointer );

    *nn_para_cpu_func_output = nn_para_cpu;

    *nn_in_dim_func_output = nn_in_dim;
    *nn_out_dim_func_output = nn_out_dim;

    return nn_layer_num;
}

int SaveKaldiCNNnet( const char *Nnet_FILE_name,
	int nn_layer_num, int nn_para_num, 
	float *nn_para_cpu,
	int *nn_in_dim, int *nn_out_dim,
	float *cnn_para_cpu, cnn_params *params)
{

    int nn_para_num_writen = 0;

    int nn_layer_count;

    int num_write_this_time;

    FILE *Nnet_file_pointer = fopen( Nnet_FILE_name, "wb" );
    if ( Nnet_file_pointer ==NULL) {
	printf("Can't CREATE Nnet FILE!\n");
	return -1;
    }

    // write '\0'
    fputc( 0, Nnet_file_pointer );

    // write 'B'
    fputc( 'B', Nnet_file_pointer );

    fprintf( Nnet_file_pointer, "<Nnet> " );
    
    // Write CNN parts
    //
    int num_splice = params->cnn_in_dim[0] / params->patch_stride[0];
    int num_sections = 1 + (params->patch_stride[0] - params->filter_section[0])/ params->filter_step[0];
    int num_patches = 1 + (params->filter_section[0] - params->patch_dim[0]) / params->patch_step[0];
    int num_filters = params->cnn_out_dim[0] / num_patches;
    int num_filters_sec = num_filters/num_sections;
    int filter_dim = num_splice * params->patch_dim[0];
    
    fprintf(Nnet_file_pointer,"<convolutionalcomponent> ");
    write_data(params->cnn_out_dim, sizeof(int), 1, Nnet_file_pointer);
    write_data(params->cnn_in_dim, sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<patchdim> ");
    write_data(params->patch_dim, sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<patchstep> ");
    write_data(params->patch_step , sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<patchstride> ");
    write_data(params->patch_stride , sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<filtersection> ");
    write_data(params->filter_section , sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<sectionstep> ");
    write_data(params->filter_step, sizeof(int), 1, Nnet_file_pointer);

    float one_f = 1.0f;

    fprintf(Nnet_file_pointer, "<learnratecoef> ");
    write_data(&one_f, sizeof(float), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<biaslearnratecoef> ");
    write_data(&one_f, sizeof(float), 1, Nnet_file_pointer);


    fprintf(Nnet_file_pointer, "<filters> FM ");
    write_data(&num_filters, sizeof(int), 1, Nnet_file_pointer);
    write_data(&filter_dim, sizeof(int), 1, Nnet_file_pointer);
	num_write_this_time = fwrite( cnn_para_cpu, 
		sizeof(float), num_filters * filter_dim,
		Nnet_file_pointer );

	if ( num_write_this_time != num_filters * filter_dim ) {
	    printf("Write Failed\n");
	    return -2;
	}

    fprintf(Nnet_file_pointer, "<bias> FV ");
    write_data(&num_filters, sizeof(int), 1, Nnet_file_pointer);
	num_write_this_time = fwrite( cnn_para_cpu + num_filters * filter_dim, 
		sizeof(float), num_filters,
		Nnet_file_pointer );

	if ( num_write_this_time != num_filters ) {
	    printf("Write Failed\n");
	    return -2;
	}

    // Write Pooling part
    fprintf(Nnet_file_pointer, "<maxpoolingcomponent> ");
    write_data(params->pool_out_dim, sizeof(int), 1, Nnet_file_pointer);
    write_data(params->cnn_out_dim, sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<poolsize> ");
    write_data(params->pool_size, sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<poolstep> ");
    write_data(params->pool_step, sizeof(int), 1, Nnet_file_pointer);

    fprintf(Nnet_file_pointer, "<poolstride> ");
    write_data(params->pool_stride, sizeof(int), 1, Nnet_file_pointer);

    // Write Sigmoid part
    fprintf(Nnet_file_pointer, "<sigmoid> ");
    write_data(params->pool_out_dim, sizeof(int), 1, Nnet_file_pointer);
    write_data(params->pool_out_dim, sizeof(int), 1, Nnet_file_pointer);

    //Write DNN parts

    for (int nn_layer_count = 0; nn_layer_count < nn_layer_num; nn_layer_count++) {

	fprintf( Nnet_file_pointer, "<affinetransform> " );

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_out_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_in_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

	// write weight
	fprintf( Nnet_file_pointer, "FM " );

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_out_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_in_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

	num_write_this_time = fwrite( nn_para_cpu + nn_para_num_writen, 
		sizeof(float), nn_out_dim[nn_layer_count] * nn_in_dim[nn_layer_count] ,
		Nnet_file_pointer );

	if ( num_write_this_time != nn_out_dim[nn_layer_count] * nn_in_dim[nn_layer_count] ) {
	    printf( "Write %d Data, Write Failed\n", num_write_this_time );
	    return -2;
	}

	nn_para_num_writen += nn_out_dim[nn_layer_count] * nn_in_dim[nn_layer_count];

	// write bias
	fprintf( Nnet_file_pointer, "FV " );

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_out_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

	num_write_this_time = fwrite( nn_para_cpu + nn_para_num_writen, 
		sizeof(float), nn_out_dim[nn_layer_count],
		Nnet_file_pointer );

	if ( num_write_this_time != nn_out_dim[nn_layer_count] ) {
	    printf( "Write %d Data, Write Failed\n", num_write_this_time );
	    return -2;
	}

	nn_para_num_writen += nn_out_dim[nn_layer_count];

	if( nn_layer_count != (nn_layer_num-1) )
	    fprintf( Nnet_file_pointer, "<sigmoid> " );
	else
	    fprintf( Nnet_file_pointer, "<softmax> " );

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_out_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

	fputc( 4, Nnet_file_pointer );

	num_write_this_time = fwrite( nn_out_dim + nn_layer_count, 
		sizeof(int), 1,
		Nnet_file_pointer );

	if ( num_write_this_time != 1 ) {
	    printf("Write Failed\n");
	    return -2;
	}

    }

    fprintf( Nnet_file_pointer, "</Nnet> " );

    fclose( Nnet_file_pointer );

    if( nn_para_num_writen != nn_para_num ) {
	printf("Para_num Mismatch in Nnet_Save!\n");
	return -3;
    }

    return 0;
}

int lbxReadFeatureTransform( const char *Nnet_FILE_name,
	int *feature_in_dim, int *feature_out_dim,
	int *splice_length,
	int **splice_index,
	float **add_shift_data, 
	float **rescale_data )
{

    char read_temp[MAX_STRING_SIZE] = "Nothing";

    int input_dim = 0;
    int output_dim = 0;
    int nnet_head = 0;
    FILE *Nnet_file_pointer = fopen( Nnet_FILE_name, "r" );
    if ( Nnet_file_pointer ==NULL) {
	printf("Can't OPEN Feature_Transform FILE!\n");
	return -1;
    }

    // read “<Nnet>”
    fscanf( Nnet_file_pointer, "%s", read_temp);
    printf("Reading: %s\n", read_temp );
    if( strcmp( read_temp, "<Nnet>" ) ) {
	printf("no <Nnet> head\n");
	nnet_head = 0;
    }else{
	nnet_head = 1;
    }

    // read “<splice>”
    if( nnet_head==1 ){
	fscanf( Nnet_file_pointer, "%s", read_temp);
    }
    if( strcmp( read_temp, "<splice>" ) && strcmp( read_temp, "<Splice>" ) ) {
	printf("Layer Type = %s\n", read_temp);
	printf("Can't support this file. Please Contact the Designers....\n");
	return -2;
    }
    fscanf( Nnet_file_pointer, "%d%d", &output_dim, &input_dim);
    // printf( "Output_dim = %d, Input_dim = %d\n", output_dim, input_dim );

    if( output_dim % input_dim ) {
	printf("Illegal NN Dim! Output_dim = %d, Input_dim = %d\n", output_dim, input_dim );
	return -3;
    }

    *feature_in_dim = input_dim;
    *feature_out_dim = output_dim;

    *splice_length = output_dim/input_dim;

    *splice_index = (int *)malloc( (*splice_length) * sizeof(int) );

    // read '['
    fscanf( Nnet_file_pointer, "%s", read_temp );

    for (int i = 0; i < (*splice_length); ++i) {
	fscanf( Nnet_file_pointer, "%d", (*splice_index)+i );
    }

    // read ']'
    fscanf( Nnet_file_pointer, "%s", read_temp );

    // read “<addshift>”
    fscanf( Nnet_file_pointer, "%s", read_temp);
    if( strcmp( read_temp, "<addshift>" ) && strcmp( read_temp, "<AddShift>" ) ) {
	printf("Layer Type = %s\n", read_temp);
	printf("Can't support this file. Please Contact the Designers....\n");
	return -2;
    }

    fscanf( Nnet_file_pointer, "%d%d", &output_dim, &input_dim);

    if( output_dim != *feature_out_dim ) {
	printf("Illegal NN Dim! Output_dim = %d, Input_dim = %d\n", output_dim, input_dim );
	return -3;
    }

    *add_shift_data = (float *)malloc( output_dim * sizeof(float) );

    // read '['
    fscanf( Nnet_file_pointer, "%s", read_temp );

    for (int i = 0; i < output_dim; ++i) {
	fscanf( Nnet_file_pointer, "%f", (*add_shift_data)+i );
    }

    // read ']'
    fscanf( Nnet_file_pointer, "%s", read_temp );

    // read “<rescale>”
    fscanf( Nnet_file_pointer, "%s", read_temp);
    if( strcmp( read_temp, "<rescale>" ) && strcmp( read_temp, "<Rescale>" ) && strcmp( read_temp, "<ReScale>" ) ) {
	printf("Layer Type = %s\n", read_temp);
	printf("Can't support this file. Please Contact the Designers....\n");
	return -2;
    }

    fscanf( Nnet_file_pointer, "%d%d", &output_dim, &input_dim);

    if( output_dim != *feature_out_dim ) {
	printf("Illegal NN Dim! Output_dim = %d, Input_dim = %d\n", output_dim, input_dim );
	return -3;
    }

    *rescale_data = (float *)malloc( output_dim * sizeof(float) );

    // read '['
    fscanf( Nnet_file_pointer, "%s", read_temp );

    for (int i = 0; i < output_dim; ++i) {
	fscanf( Nnet_file_pointer, "%f", (*rescale_data)+i );
    }

    // read ']'
    fscanf( Nnet_file_pointer, "%s", read_temp );

    // read '</Nnet>'
    if( nnet_head==1 ){
	fscanf( Nnet_file_pointer, "%s", read_temp );
	if ( strcmp( read_temp, "</Nnet>" ) ) {
	    printf("Illegal End of File: %s\n", read_temp );
	    return -4;
	}
    }
    fclose( Nnet_file_pointer );

    return 0;
}


