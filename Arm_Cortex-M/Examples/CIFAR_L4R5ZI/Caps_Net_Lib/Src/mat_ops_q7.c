#include "mat_ops_q7.h"
#include "arm_math.h"
#include "arm_nnsupportfunctions.h"
#include "matrix_q7_backend.h"


void mat_mult_q7_trb(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst, q7_t *buffer)
{
	transpose_matrix_q7(pSrcB->pData, pSrcB->numRows, pSrcB->numCols, buffer);
	
	/* Variables for usage in following multiplication process */
	q7_t *p_in_a, *p_in_b;
	uint16_t num_cols_a = pSrcA->numCols;
	uint16_t num_cols_b = pSrcB->numCols;
	uint16_t row = pSrcA->numRows;
	uint16_t col, col_cnt;
	int32_t sum;
	uint16_t i = 0;
	q7_t *px = pDst->pData;
	int32_t div = 1 << out_rshift;
	
	/* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
	/* row loop */
	while(row > 0)
	{
		/* For every row wise process, column loop counter is to be initiated */
		col = num_cols_b;
		/* For every row wise process, pIn2 pointer is set to starting address of transposed pSrcB data */
		p_in_b = buffer;

		/* column loop */
		do
		{
			sum = 0;
			/* Initiate pointer pInA to point to starting address of column being processed */
			p_in_a = pSrcA->pData + i;
			
			col_cnt = num_cols_a;

			while (col_cnt > 0U)
			{
				sum += *p_in_a++ * *p_in_b++;
				col_cnt--;
			}

			/* Saturate and store result in destination buffer */
			*px = (q7_t) (__SSAT((sum/div), 8));
			px++;

			/* Decrement column loop counter */
			col--;
			
		} while (col > 0U);

		i = i + num_cols_a;
		row--;
	}
}


void mat_mult_q7_simd(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst, q15_t *buffer)
{
	matrix_q7_to_q15_transposed(pSrcB->pData, pSrcB->numRows, pSrcB->numCols, buffer);

	/* Variables for usage in following multiplication process */
	q7_t *p_in_a;
	q15_t *p_in_b;
	uint16_t num_cols_a = pSrcA->numCols;
	uint16_t num_cols_b = pSrcB->numCols;
    uint16_t row = pSrcA->numRows;
	uint16_t col, col_cnt;
	int32_t sum;
    uint16_t i = 0;
	q7_t *px = pDst->pData;
	int32_t div = 1 << out_rshift;

	/* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    while(row > 0)
    {
        /* For every row wise process, column loop counter is to be initiated */
        col = num_cols_b;

        /* For every row wise process, pIn2 pointer is set to starting address of transposed pSrcB data */
        p_in_b = buffer;

      	/* column loop */
		do
		{
			/* Set variable sum, that acts as accumulator, to zero */
			sum = 0;

			/* Initiate pointer pInA to point to starting address of column being processed */
			p_in_a = pSrcA->pData + i;

			/* Apply loop unrolling and compute 2 MACs simultaneously. */
			col_cnt = num_cols_a >> 2U;

			/* matrix multiplication */
			while(col_cnt > 0U)
			{
                q31_t in_a_11, in_a_12, in_b_11, in_b_12;

				/* Read and expand one q7 word into two q15 words */
				p_in_a = read_and_pad_reordered(p_in_a, &in_a_11, &in_a_12);
                /* Read and expand one q7 word into two q15 words */
				in_b_11 = read_q15x2_ia((q15_t**)&p_in_b);
				in_b_12 = read_q15x2_ia((q15_t**)&p_in_b);

                sum = __SMLAD(in_a_11, in_b_11, sum);
                sum = __SMLAD(in_a_12, in_b_12, sum);

				/* Decrement loop counter */
				col_cnt--;
			}

			/* process remaining column samples */
			col_cnt = num_cols_a % 0x4U;

			while (col_cnt > 0U)
			{
				/* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */
				sum += *p_in_a++ * *p_in_b++;

				/* Decrement loop counter */
				col_cnt--;
			}

			/* Saturate and store result in destination buffer */
			*px = (q7_t) (__SSAT((sum/div), 8));
			px++;

			/* Decrement column loop counter */
			col--;

		} while (col > 0U);

		i = i + num_cols_a;

		/* Decrement row loop counter */
		row--;
	}
}


void mat_add_q7(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst)
{
	uint16_t m_size = pSrcA->numRows * pSrcA->numCols;
	q7_t *p_in_a = pSrcA->pData;
	q7_t *p_in_b = pSrcB->pData;
	q7_t *px = pDst->pData;
	int32_t val;
	int32_t div = 1 << out_rshift;
	
	for(uint16_t i=0; i<m_size; i++){
		val = *p_in_a + *p_in_b;
		*px = (q7_t) (__SSAT((val / div), 8));
		px++;
		p_in_a++;
		p_in_b++;
	}
}
