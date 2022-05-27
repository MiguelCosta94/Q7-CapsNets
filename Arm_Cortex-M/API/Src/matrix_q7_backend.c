#include "matrix_q7_backend.h"
#include "arm_math.h"


void transpose_matrix_q7(q7_t *in_matrix, uint16_t rows, uint16_t columns, q7_t *out_matrix)
{
	q7_t *px;				/* Temporary output data matrix pointer */
	uint32_t col, i=0U, in, row = rows;
	
	do
	{
		/* The pointer px is set to starting address of column being processed */
		px = out_matrix + i;

		/* Apply loop unrolling and exchange columns with row elements */
		col = columns >> 2U;

		/* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
		 ** a second loop below computes the remaining 1 to 3 samples. */
		while (col > 0U)
		{
			/* Read four elements from row */
			in = read_q7x4_ia ((q7_t **) &in_matrix);

			/* Unpack and store first element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q7_t) (in & (q31_t) 0x000000ff);
#else
			*px = (q7_t) ((in & (q31_t) 0xff000000) >> 24);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Unpack and store second element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q7_t) ((in & (q31_t) 0x0000ff00) >> 8);
#else
			*px = (q7_t) ((in & (q31_t) 0x00ff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;
				
			/* Unpack and store thrid element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q7_t) ((in & (q31_t) 0x00ff0000) >> 16);
#else
			*px = (q7_t) ((in & (q31_t) 0x0000ff00) >> 8);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Unpack and store fourth element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q7_t) ((in & (q31_t) 0xff000000) >> 24);
#else
			*px = (q7_t) (in & (q31_t) 0x000000ff);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Decrement column loop counter */
			col--;
		}

		/* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
		 ** No loop unrolling is used. */
		col = columns % 0x4U;

		while (col > 0U)
		{
			/* Read and store input element in destination */
			*px = *in_matrix++;
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;
			/* Decrement column loop counter */
			col--;
		}

		i++;
		/* Decrement row loop counter */
		row--;
	} while (row > 0U);
}


void matrix_q7_to_q15_transposed(q7_t *in_matrix, uint16_t rows, uint16_t columns, q15_t *out_matrix)
{
	q15_t *px;				/* Temporary output data matrix pointer */
	uint32_t col, i=0U, in, row = rows;
	
	do
	{
		/* The pointer px is set to starting address of column being processed */
		px = out_matrix + i;

		/* Apply loop unrolling and exchange columns with row elements */
		col = columns >> 2U;

		/* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
		 ** a second loop below computes the remaining 1 to 3 samples. */
		while (col > 0U)
		{
			/* Read four elements from row */
			in = read_q7x4_ia ((q7_t **) &in_matrix);

			/* Unpack and store first element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q15_t) (in & (q31_t) 0x000000ff);
#else
			*px = (q15_t) ((in & (q31_t) 0xff000000) >> 24);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Unpack and store second element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q15_t) ((in & (q31_t) 0x0000ff00) >> 8);
#else
			*px = (q15_t) ((in & (q31_t) 0x00ff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;
				
			/* Unpack and store thrid element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q15_t) ((in & (q31_t) 0x00ff0000) >> 16);
#else
			*px = (q15_t) ((in & (q31_t) 0x0000ff00) >> 8);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Unpack and store fourth element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
			*px = (q15_t) ((in & (q31_t) 0xff000000) >> 24);
#else
			*px = (q15_t) (in & (q31_t) 0x000000ff);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Decrement column loop counter */
			col--;
		}

		/* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
		 ** No loop unrolling is used. */
		col = columns % 0x4U;

		while (col > 0U)
		{
			/* Read and store input element in destination */
			*px = *in_matrix++;
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;
			/* Decrement column loop counter */
			col--;
		}

		i++;
		/* Decrement row loop counter */
		row--;
	} while (row > 0U);
}