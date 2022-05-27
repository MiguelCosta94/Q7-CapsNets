#ifndef MATH_BACKEND_H
#define MATH_BACKEND_H

#include "matrix_q7_backend.h"
#include "pmsis.h"


uint32_t vector_sq_norm_q7(int8_t *input, uint16_t length);
uint32_t int_sqrt(uint32_t s);

#endif
