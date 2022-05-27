#ifndef SOFTMAX_Q7_H
#define SOFTMAX_Q7_H

#include "pmsis.h"

void softmax_q7(int8_t *vec_in, const uint16_t dim_vec, int8_t *p_out);

#endif