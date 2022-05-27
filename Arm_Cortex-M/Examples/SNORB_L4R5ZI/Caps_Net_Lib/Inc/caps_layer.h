#ifndef CAPS_LAYER_H
#define CAPS_LAYER_H

#include "arm_math.h"

/* Functions */
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
									uint16_t squash_out_qn, q7_t *output, uint16_t out_dim_x, uint16_t out_dim_y, q15_t *buffer_aux);

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
									uint16_t squash_out_qn, q7_t *output, uint16_t out_dim_x, uint16_t out_dim_y, q15_t *buffer_aux);

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
					q7_t *output, q7_t *buffer_inputs_hat, q7_t *buffer_b, q7_t *buffer_c, q7_t *buffer_aux);

void capsule_length_q7(q7_t *input, uint16_t num_caps, uint16_t dim_caps, q7_t *output);

#endif
