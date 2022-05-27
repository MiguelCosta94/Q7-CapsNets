#include "caps_layer.h"
#include "arm_nnfunctions.h"
#include "arm_math.h"
#include "math_backend.h"
#include "mat_ops_q7.h"
#include "matrix_q7_backend.h"
#include "stdlib.h"

/*****************************************************************************************/
/*************************** Declaration of auxiliar functions ***************************/
void calc_inputs_hat(q7_t *input, q7_t *wt, uint16_t num_caps, uint16_t dim_caps, uint16_t input_num_caps,
						uint16_t input_dim_caps, uint16_t out_rshift, q7_t *inputs_hat, q7_t *buffer_aux);

void calc_coupling_coefs(q7_t *b, uint16_t num_caps, uint16_t input_num_caps, q7_t *c, q7_t *buffer_aux);

void calc_caps_output(q7_t *inputs_hat, q7_t *c, uint16_t num_caps, uint16_t input_num_caps, uint16_t dim_caps,
						uint16_t out_rshift, uint16_t squash_in_qn, uint16_t squash_out_qn, q7_t *output, q7_t *buffer_aux);

void calc_agreement_w_prev_caps(q7_t *inputs_hat, q7_t *output, uint16_t num_caps, uint16_t dim_caps, uint16_t input_num_caps,
								uint16_t b_inst_shift, uint16_t b_new_shift, q7_t *b, q7_t *b_inst_buffer, q7_t *buffer_aux);

void squash_q7(q7_t *matrix, uint16_t rows, uint16_t columns, uint16_t in_qn, uint16_t out_qn);
/*****************************************************************************************/
/************************* Implementation of auxiliar functions **************************/
void calc_inputs_hat(q7_t *input, q7_t *wt, uint16_t num_caps, uint16_t dim_caps, uint16_t input_num_caps,
						uint16_t input_dim_caps, uint16_t out_rshift, q7_t *inputs_hat, q7_t *buffer_aux)
{
	// Compute `weights * inputs
	// input.shape=[num_capsule, input_num_capsule, input_dim_capsule, 1]
	// wt.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
	// Regard the 2 inner dimensions as valid matrix multiplication dimensions and outer dimensions as batch size
	// then matmul: [dim_capsule, input_dim_capsule] x [input_dim_capsule, 1] -> [dim_capsule, 1]
	// inputs_hat.shape = [num_capsule, input_num_capsule, dim_capsule]
	q7_t *input_cpy = input;
	
	for(uint16_t i=0; i<num_caps; i++){
		for(uint16_t j=0; j<input_num_caps; j++){
			matrix_instance_q7 wt_matrix = {.numRows=dim_caps, .numCols=input_dim_caps, .pData=wt};
			matrix_instance_q7 inputs_matrix = {.numRows=input_dim_caps, .numCols=1, .pData=input_cpy};
			matrix_instance_q7 inputs_hat_matrix = {.numRows=dim_caps, .numCols=1, .pData=inputs_hat};
			
			mat_mult_q7_trb(&wt_matrix, &inputs_matrix, out_rshift, &inputs_hat_matrix, buffer_aux);
			          
			wt += wt_matrix.numRows * wt_matrix.numCols;
			input_cpy += inputs_matrix.numRows * inputs_matrix.numCols;
			inputs_hat += inputs_hat_matrix.numRows * inputs_hat_matrix.numCols;
		}
		input_cpy = input;
	}
}


void calc_coupling_coefs(q7_t *b, uint16_t num_caps, uint16_t input_num_caps, q7_t *c, q7_t *buffer_aux)
{
	q7_t *b_trans = buffer_aux;
	q7_t *c_trans = c;
	
	transpose_matrix_q7(b, num_caps, input_num_caps, b_trans);
	
	for(uint16_t j=0; j<input_num_caps; j++){
		arm_softmax_q7(b_trans, num_caps, c_trans);			//C is transposed
		b_trans += num_caps;
		c_trans += num_caps;
	}
	
	c_trans = c;
	q7_t *c_not_trans = buffer_aux;
	
	transpose_matrix_q7(c_trans, input_num_caps, num_caps, c_not_trans);
	memcpy(c, c_not_trans, num_caps * input_num_caps);		//C is not transposed
}


void calc_caps_output(q7_t *inputs_hat, q7_t *c, uint16_t num_caps, uint16_t input_num_caps, uint16_t dim_caps,
						uint16_t out_rshift, uint16_t squash_in_qn, uint16_t squash_out_qn, q7_t *output, q7_t *buffer_aux)
{
	q7_t *output_cpy = output;
	
	for(uint16_t j=0; j<num_caps; j++){
		matrix_instance_q7 c_matrix = {.numRows=1, .numCols=input_num_caps, .pData=c};
		matrix_instance_q7 inputs_hat_matrix = {.numRows=input_num_caps, .numCols=dim_caps, .pData=inputs_hat};
		matrix_instance_q7 outputs_matrix = {.numRows=1, .numCols=dim_caps, .pData=output_cpy};
		
		mat_mult_q7_trb(&c_matrix, &inputs_hat_matrix, out_rshift, &outputs_matrix, buffer_aux);
		
		c += c_matrix.numRows * c_matrix.numCols;
		inputs_hat += inputs_hat_matrix.numRows * inputs_hat_matrix.numCols;
		output_cpy += outputs_matrix.numRows * outputs_matrix.numCols;
	}
	
	squash_q7(output, num_caps, dim_caps, squash_in_qn, squash_out_qn);
}


void calc_agreement_w_prev_caps(q7_t *inputs_hat, q7_t *output, uint16_t num_caps, uint16_t dim_caps, uint16_t input_num_caps,
								uint16_t b_inst_shift, uint16_t b_new_shift, q7_t *b, q7_t *b_inst_buffer, q7_t *buffer_aux)
{
	// outputs.shape = [num_capsule, dim_capsule, 1]
	// inputs_hat.shape=[num_capsule, input_num_capsule, dim_capsule]
	// The first dimension as `batch` dimension,
	// then matmul: [input_num_capsule, dim_capsule] x [dim_capsule, 1]
	// b.shape=[num_capsule, input_num_capsule]
	q7_t *b_inst = b_inst_buffer;
	
	for(uint16_t j=0; j<num_caps; j++){
		matrix_instance_q7 inputs_hat_matrix = {.numRows=input_num_caps, .numCols=dim_caps, .pData=inputs_hat};
		matrix_instance_q7 outputs_matrix = {.numRows=dim_caps, .numCols=1, .pData=output};
		matrix_instance_q7 inst_b_matrix = {.numRows=input_num_caps, .numCols=1, .pData=b_inst};
		
		mat_mult_q7_trb(&inputs_hat_matrix, &outputs_matrix, b_inst_shift, &inst_b_matrix, buffer_aux);
			
		inputs_hat += inputs_hat_matrix.numRows * inputs_hat_matrix.numCols;
		output += outputs_matrix.numRows * outputs_matrix.numCols;
		b_inst += inst_b_matrix.numRows * inst_b_matrix.numCols;
	}
	
	matrix_instance_q7 inst_b_matrix = {.numRows=num_caps, .numCols=input_num_caps, .pData=b_inst_buffer};
	matrix_instance_q7 old_b_matrix = {.numRows=num_caps, .numCols=input_num_caps, .pData=b};
	matrix_instance_q7 new_b_matrix = {.numRows=num_caps, .numCols=input_num_caps, .pData=b};

	mat_add_q7(&inst_b_matrix, &old_b_matrix, b_new_shift, &new_b_matrix);
}


void squash_q7(q7_t *matrix, uint16_t rows, uint16_t columns, uint16_t in_qn, uint16_t out_qn)
{
	int32_t val, squared_norm, norm, bias;
	uint16_t row;
	
	for(uint16_t i=0; i < rows; i++)
	{
		row = i * columns;
		
		squared_norm = vector_sq_norm_q7(&(matrix[row]), columns);
		norm = int_sqrt(squared_norm);
		
		squared_norm = squared_norm >> in_qn;
		norm = norm << (out_qn - in_qn);
		bias = 1 << in_qn;
		
		for(uint16_t j=0; j < columns; j++)
		{
			val = ((int32_t)(matrix[row+j]) * norm) / (squared_norm + bias);
			matrix[row+j] = (q7_t) (__SSAT(val, 8));
		}
	}
}

/*****************************************************************************************/
/**************************** Implementation of main functions ***************************/
void primary_capsule_layer_q7_basic(q7_t *input,
									uint16_t in_dim_x,
									uint16_t in_dim_y,
									uint16_t ch_dim,
									q7_t *wt,
									uint16_t num_caps,
									uint16_t dim_caps,
									uint16_t kernel_size_x,
									uint16_t kernel_size_y,
									uint16_t padding_x,
									uint16_t padding_y, 
									uint16_t stride_x,
									uint16_t stride_y,
									q7_t *bias,
									uint16_t bias_lshift,
									uint16_t out_rshift,
									uint16_t squash_in_qn,
									uint16_t squash_out_qn, q7_t *output, uint16_t out_dim_x, uint16_t out_dim_y, q15_t *buffer_aux)
{	
	arm_convolve_HWC_q7_basic_nonsquare(input, in_dim_x, in_dim_y, ch_dim, wt, dim_caps * num_caps, kernel_size_x, kernel_size_y,
										padding_x, padding_y, stride_x, stride_y, bias, bias_lshift, out_rshift, output, out_dim_x,
										out_dim_y, buffer_aux, NULL);
	
	squash_q7(output, out_dim_x * out_dim_y * num_caps, dim_caps, squash_in_qn, squash_out_qn);
}


void primary_capsule_layer_q7_fast(q7_t *input,
									uint16_t in_dim_x,
									uint16_t in_dim_y,
									uint16_t ch_dim,
									q7_t *wt,
									uint16_t num_caps,
									uint16_t dim_caps,
									uint16_t kernel_size_x,
									uint16_t kernel_size_y,
									uint16_t padding_x,
									uint16_t padding_y, 
									uint16_t stride_x,
									uint16_t stride_y,
									q7_t *bias,
									uint16_t bias_lshift,
									uint16_t out_rshift,
									uint16_t squash_in_qn,
									uint16_t squash_out_qn, q7_t *output, uint16_t out_dim_x, uint16_t out_dim_y, q15_t *buffer_aux)
{
	arm_convolve_HWC_q7_fast_nonsquare(input, in_dim_x, in_dim_y, ch_dim, wt, dim_caps * num_caps, kernel_size_x, kernel_size_y,
									padding_x, padding_y, stride_x, stride_y, bias, bias_lshift, out_rshift, output, out_dim_x,
									out_dim_y, buffer_aux, NULL);
	
	squash_q7(output, out_dim_x * out_dim_y * num_caps, dim_caps, squash_in_qn, squash_out_qn);
}


void capsule_layer_q7(q7_t *input,
						uint16_t num_caps,
						uint16_t dim_caps,
						uint16_t input_num_caps,
						uint16_t input_dim_caps,
						uint16_t routings,
						q7_t *wt,
						uint16_t input_hat_rshift,
						uint16_t *out_rshift,
						uint16_t *b_inst_shift,
						uint16_t *b_new_shift,
						uint16_t *squash_in_qn,
						uint16_t *squash_out_qn,
						q7_t *output, q7_t *buffer_inputs_hat, q7_t *buffer_b, q7_t *buffer_c, q7_t *buffer_aux)
{
	memset(buffer_b, 0, num_caps * input_num_caps);
	calc_inputs_hat(input, wt, num_caps, dim_caps, input_num_caps, input_dim_caps, input_hat_rshift, buffer_inputs_hat, buffer_aux);
	
	for(uint16_t i=0; i<routings; i++){
		calc_coupling_coefs(buffer_b, num_caps, input_num_caps, buffer_c, buffer_aux);

		calc_caps_output(buffer_inputs_hat, buffer_c, num_caps, input_num_caps, dim_caps,
						out_rshift[i], squash_in_qn[i], squash_out_qn[i], output, buffer_aux);
		
		if(i<routings-1){
			calc_agreement_w_prev_caps(buffer_inputs_hat, output, num_caps, dim_caps, input_num_caps,
										b_inst_shift[i], b_new_shift[i], buffer_b, buffer_c, buffer_aux);
		}
	}
}


void capsule_length_q7(q7_t *input, uint16_t num_caps, uint16_t dim_caps, q7_t *output)
{
	uint32_t norm;
	
	for(uint16_t i=0; i<num_caps; i++){
		norm = vector_sq_norm_q7(&(input[i*dim_caps]), dim_caps);
		norm = int_sqrt(norm);
		output[i] = (q7_t) (__SSAT(norm, 8));
	}
}
