#ifndef MATRIX_Q7_BACKEND_H
#define MATRIX_Q7_BACKEND_H

#include "pmsis.h"

/* Instance structure for int8 matrix */
typedef struct
{
	uint16_t numRows;     /**< number of rows of the matrix.     */
	uint16_t numCols;     /**< number of columns of the matrix.  */
	int8_t *pData;          /**< points to the data of the matrix. */
} matrix_instance_q7;

void transpose_matrix_q7(int8_t *in_matrix, uint16_t rows, uint16_t columns, int8_t *out_matrix);

#endif
