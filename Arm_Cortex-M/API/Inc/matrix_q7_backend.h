#ifndef MATRIX_UTILS_Q7_H
#define MATRIX_UTILS_Q7_H

#include "arm_math.h"

/* Instance structure for q7 matrix */
typedef struct
{
	uint16_t numRows;     /**< number of rows of the matrix.     */
	uint16_t numCols;     /**< number of columns of the matrix.  */
	q7_t *pData;          /**< points to the data of the matrix. */
} matrix_instance_q7;

void transpose_matrix_q7(q7_t *in_matrix, uint16_t rows, uint16_t columns, q7_t *out_matrix);
void matrix_q7_to_q15_transposed(q7_t *in_matrix, uint16_t rows, uint16_t columns, q15_t *out_matrix);

#endif
