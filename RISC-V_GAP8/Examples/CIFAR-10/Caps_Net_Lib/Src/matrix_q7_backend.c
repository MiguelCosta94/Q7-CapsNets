#include "matrix_q7_backend.h"
#include "pmsis.h"


#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))


void transpose_matrix_q7(int8_t *in_matrix, uint16_t rows, uint16_t columns, int8_t *out_matrix)
{
	int log2_core = log2(NUM_CORES);
	int core_id = pi_core_id();
    /*chunks are built along the spatial dimension of the OFM */
    int chunk = (rows >> log2_core) + ((rows & (NUM_CORES - 1)) != 0);

    /* defining the specific rows computed by each core */
    int start_row = min(chunk * core_id, rows);
    int stop_row = min(start_row + chunk, rows);
    int eff_chunk = stop_row - start_row;

	int8_t *px;				/* Temporary output data matrix pointer */
	int8_t *p_in = in_matrix + (start_row * columns);
	uint16_t col, i = start_row;
	int32_t in;
	
	while(eff_chunk>0)
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
			memcpy(&in, p_in, sizeof(in));

			/* Unpack and store first element in destination */
			*px = (int8_t) (in & (int32_t) 0x000000ff);
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Unpack and store second element in destination */
			*px = (int8_t) ((in & (int32_t) 0x0000ff00) >> 8);
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;
				
			/* Unpack and store thrid element in destination */
			*px = (int8_t) ((in & (int32_t) 0x00ff0000) >> 16);
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Unpack and store fourth element in destination */
			*px = (int8_t) ((in & (int32_t) 0xff000000) >> 24);
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;

			/* Decrement column loop counter */
			col--;
			/* Set pointer to the next four row elements */
			p_in += 4;
		}

		/* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
		 ** No loop unrolling is used. */
		col = columns % 0x4U;

		while (col > 0U)
		{
			/* Read and store input element in destination */
			*px = *p_in++;
			/* Update pointer px to point to next row of transposed matrix */
			px += rows;
			/* Decrement column loop counter */
			col--;
		}

		i++;
		/* Decrement row loop counter */
		eff_chunk--; 
	}

	pi_cl_team_barrier(0);
}