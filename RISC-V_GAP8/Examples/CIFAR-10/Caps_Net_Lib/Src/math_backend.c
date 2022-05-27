#include "math_backend.h"
#include "pmsis.h"


uint32_t vector_sq_norm_q7(int8_t *input, uint16_t length)
{
	uint32_t sq_norm = 0;
	
	for(uint16_t i=0; i<length; i++)
	{
		sq_norm += (uint32_t)((int32_t)input[i] * (int32_t)input[i]);
	}
	
	return sq_norm;
}


// Square root of integer
// Newton's method
uint32_t int_sqrt(uint32_t s)
{
	if(s==1)
	{
		return 1;
	}
	
	uint32_t x0 = s >> 1;					// Initial estimate
	uint32_t x1 = (x0 + s / x0) >> 1;		// Update
		
	while (x1 < x0)							// This also checks for cycle
	{
		x0 = x1;
		x1 = (x0 + s / x0) >> 1;
	}
	
	return x0;
}