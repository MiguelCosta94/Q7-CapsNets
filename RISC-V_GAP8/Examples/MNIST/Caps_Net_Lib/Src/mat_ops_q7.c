#include "mat_ops_q7.h"
#include "matrix_q7_backend.h"
#include "pmsis.h"


#define log2(x) __builtin_pulp_fl1(x)
#define min(a,b) ((a)<(b)?(a):(b))


void mat_mult_q7(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst)
{
	int log2_core = log2(NUM_CORES);
    /*chunks are built along the spatial dimension of the OFM */
    int chunk = (pSrcA->numRows >> log2_core) + ((pSrcA->numRows & (NUM_CORES - 1)) != 0);
    /* defining the specific rows computed by each core */
	int core_id = pi_core_id();
    int start_row = min(chunk * core_id, pSrcA->numRows);
    int stop_row = min(start_row + chunk, pSrcA->numRows);

	uint16_t c1 = pSrcA->numCols;
	uint16_t r2 = pSrcB->numRows;
	uint16_t c2 = pSrcB->numCols;
	int8_t *first = pSrcA->pData;
	int8_t *second = pSrcB->pData;
	int8_t *res = pDst->pData;
	int32_t div = 1 << out_rshift;
	int32_t val;
	
	// Multiplying first and second matrices and storing it in result
	for (uint16_t i = start_row; i < stop_row; i++) {
		for (uint16_t j = 0; j < c2; j++) {
			val = 0;

			for (uint16_t k = 0; k < c1; k++) {
				val += (int32_t)first[(i * c1) + k] * (int32_t)second[(k * c2) + j];
			}
			res[(i * c2) + j] = (int8_t) (__builtin_pulp_clip_r((val/div), 127));
		}
	}
}


void mat_mult_q7_trb(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst, int8_t *buffer)
{
	transpose_matrix_q7(pSrcB->pData, pSrcB->numRows, pSrcB->numCols, buffer);

	/* Variables for usage in following multiplication process */
	int8_t *p_in_a, *p_in_b;
	uint16_t num_cols_a = pSrcA->numCols;
	uint16_t num_cols_b = pSrcB->numCols;
	uint16_t row = pSrcA->numRows;
	uint16_t col, col_cnt;
	int32_t sum;
	int32_t div = 1 << out_rshift;

    int log2_core = log2(NUM_CORES);
    /*chunks are built along the spatial dimension of the OFM */
    int chunk = (pSrcA->numRows >> log2_core) + ((pSrcA->numRows & (NUM_CORES - 1)) != 0);

    /* defining the specific rows computed by each core */
	int core_id = pi_core_id();
    int start_row = min(chunk * core_id, pSrcA->numRows);
    int stop_row = min(start_row + chunk, pSrcA->numRows);
    int eff_chunk = stop_row - start_row;
    uint16_t i = start_row * num_cols_a;
	int8_t *px = pDst->pData + (start_row * num_cols_b);
	
	/* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    while(eff_chunk>0)
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
			*px = (int8_t) (__builtin_pulp_clip_r((sum/div), 127));
			px++;

			/* Decrement column loop counter */
			col--;
			
		} while (col > 0U);

		i = i + num_cols_a;
		eff_chunk--;
	}

	pi_cl_team_barrier(0);
}


void mat_mult_q7_simd(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst, int8_t *buffer)
{
	transpose_matrix_q7(pSrcB->pData, pSrcB->numRows, pSrcB->numCols, buffer);

	/* Variables for usage in following multiplication process */
	int8_t *p_in_a, *p_in_b;
	uint16_t num_cols_a = pSrcA->numCols;
	uint16_t num_cols_b = pSrcB->numCols;
    uint16_t row = pSrcA->numRows;
	uint16_t col, col_cnt;
	int32_t sum;

    int log2_core = log2(NUM_CORES);
    /*chunks are built along the spatial dimension of the OFM */
    int chunk = (pSrcA->numRows >> log2_core) + ((pSrcA->numRows & (NUM_CORES - 1)) != 0);

    /* defining the specific rows computed by each core */
	int core_id = pi_core_id();
    int start_row = min(chunk * core_id, pSrcA->numRows);
    int stop_row = min(start_row + chunk, pSrcA->numRows);
    int eff_chunk = stop_row - start_row;
    uint16_t i = start_row * num_cols_a;
	int8_t *px = pDst->pData + (start_row * num_cols_b);
	int32_t div = 1 << out_rshift;

	/* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    while(eff_chunk>0)
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
				v4s in_a = *((v4s*) p_in_a);
				v4s in_b = *((v4s*) p_in_b);

				sum = __builtin_pulp_sdotsp4(in_a, in_b, sum);
				p_in_a+=4;
				p_in_b+=4;

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
			*px = (int8_t) (__builtin_pulp_clip_r((sum / div), 127));
			px++;

			/* Decrement column loop counter */
			col--;

		} while (col > 0U);

		i = i + num_cols_a;

		/* Decrement row loop counter */
		eff_chunk--;
	}

	pi_cl_team_barrier(0);
}


void mat_add_q7(matrix_instance_q7 *pSrcA, matrix_instance_q7 *pSrcB, uint16_t out_rshift, matrix_instance_q7 *pDst)
{
    uint16_t num_rows = pSrcA->numRows;
    uint16_t num_cols = pSrcA->numCols;
    int8_t *p_in_a, *p_in_b, *px;
    uint16_t col, col_cnt;

    int log2_core = log2(NUM_CORES);
    /*chunks are built along the rows */
    int chunk = (num_rows >> log2_core) + ((num_rows & (NUM_CORES - 1)) != 0);

    /* defining the specific rows computed by each core */
	int core_id = pi_core_id();
    int start_row = min(chunk * core_id, num_rows);
    int stop_row = min(start_row + chunk, num_rows);
    int eff_chunk = stop_row - start_row;

    uint16_t i = start_row * num_cols;
	int32_t div = 1 << out_rshift;
	
    /* The following loop performs the addition of each row in matrix A with each row in Matrix B */
    /* row loop */
    while(eff_chunk>0)
    {
        /* Initiate pointer pInA to point to starting address of row being processed */
        p_in_a = pSrcA->pData + i;

        /* Initiate pointer pInB to point to starting address of row being processed */
        p_in_b = pSrcB->pData + i;

        /* Initiate pointer px to point to starting address of row being processed */
        px = pDst->pData + i;

        /* Apply loop unrolling */
        col_cnt = num_cols >> 2U;

        /* matrix multiplication */
        while(col_cnt > 0U)
        {
            v4s in_a = *((v4s*) p_in_a);
            v4s in_b = *((v4s*) p_in_b);

            v4s sum = __builtin_pulp_add4(in_a, in_b);

            /* Saturate and store result in destination buffer */
            *px++ = (int8_t) (__builtin_pulp_clip_r((sum[0] / div), 127));
            *px++ = (int8_t) (__builtin_pulp_clip_r((sum[1] / div), 127));
            *px++ = (int8_t) (__builtin_pulp_clip_r((sum[2] / div), 127));
            *px++ = (int8_t) (__builtin_pulp_clip_r((sum[3] / div), 127));

            p_in_a+=4;
            p_in_b+=4;

            /* Decrement loop counter */
            col_cnt--;
        }

        /* process remaining column samples */
        col_cnt = num_cols % 0x4U;

        while (col_cnt > 0U)
        {
            int s_sum = *p_in_a++ + *p_in_b++;
            *px++ = (int8_t) (__builtin_pulp_clip_r((s_sum / div), 127));

            /* Decrement loop counter */
            col_cnt--;
        }

        i = i + num_cols;

		/* Decrement row loop counter */
		eff_chunk--;
	}

    pi_cl_team_barrier(0);
}
