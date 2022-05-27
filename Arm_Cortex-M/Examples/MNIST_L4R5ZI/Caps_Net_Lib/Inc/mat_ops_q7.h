#ifndef MAT_OPS_Q7_H
#define MAT_OPS_Q7_H

#include "arm_math.h"
#include "matrix_q7_backend.h"

void mat_mult_q7_trb(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst, q7_t *buffer);
void mat_mult_q7_simd(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst, q15_t *buffer);
void mat_add_q7(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst);

#endif
