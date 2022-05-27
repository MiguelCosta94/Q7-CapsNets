#ifndef CAPS_LAYER_H
#define CAPS_LAYER_H

#include "pmsis.h"

/* Functions */
void primary_capsule_layer_Ho_parallel_q7(int8_t *input,
										uint16_t in_dim_x,
										uint16_t in_dim_y,
										uint16_t ch_dim,
										int8_t *wt,
										uint16_t num_caps,
										uint16_t dim_caps,
										uint16_t kernel_size_x,
										uint16_t kernel_size_y,
										uint16_t padding_y_top,
										uint16_t padding_y_bottom,
										uint16_t padding_x_left,
										uint16_t padding_x_right,
										uint16_t stride_x,
										uint16_t stride_y,
										int8_t *bias,
										uint16_t bias_lshift,
										uint16_t out_rshift,
										uint16_t squash_in_qn,
										uint16_t squash_out_qn, int8_t *output, uint16_t out_dim_x, uint16_t out_dim_y, int8_t *buffer_aux);

void primary_capsule_layer_HoWo_parallel_q7(int8_t *input,
										uint16_t in_dim_x,
										uint16_t in_dim_y,
										uint16_t ch_dim,
										int8_t *wt,
										uint16_t num_caps,
										uint16_t dim_caps,
										uint16_t kernel_size_x,
										uint16_t kernel_size_y,
										uint16_t padding_y_top,
										uint16_t padding_y_bottom,
										uint16_t padding_x_left,
										uint16_t padding_x_right,
										uint16_t stride_x,
										uint16_t stride_y,
										int8_t *bias,
										uint16_t bias_lshift,
										uint16_t out_rshift,
										uint16_t squash_in_qn,
										uint16_t squash_out_qn, int8_t *output, uint16_t out_dim_x, uint16_t out_dim_y, int8_t *buffer_aux);

void capsule_layer_q7(int8_t *input,
						uint16_t num_caps,
						uint16_t dim_caps,
						uint16_t input_num_caps,
						uint16_t input_dim_caps,
						uint16_t routings,
						int8_t *wt,
						uint16_t input_hat_rshift,
						uint16_t *out_rshift,
						uint16_t *b_inst_shift,
						uint16_t *b_new_shift,
						uint16_t *squash_in_qn,
						uint16_t *squash_out_qn,
						int8_t *output, int8_t *buffer_inputs_hat, int8_t *buffer_b, int8_t *buffer_c, int8_t *buffer_aux);

void capsule_length_q7(int8_t *input, uint16_t num_caps, uint16_t dim_caps, int8_t *output);

#endif
