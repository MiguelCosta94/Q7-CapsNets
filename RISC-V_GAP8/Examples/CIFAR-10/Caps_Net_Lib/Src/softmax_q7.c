#include "softmax_q7.h"
#include "pmsis.h"


void softmax_q7(int8_t *vec_in, const uint16_t dim_vec, int8_t *p_out)
{
    int32_t sum;
    int16_t i;
    uint8_t shift;
    int16_t base;
    base = -128;

    /* We first search for the maximum */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    /*
     * So the base is set to max-8, meaning
     * that we ignore really small values.
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - (1 << 3);

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        shift = (uint8_t)__builtin_pulp_clipu_r(vec_in[i] - base, 7);
        sum += 0x1 << shift;
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = (1 << 20) / sum;

    for (i = 0; i < dim_vec; i++)
    {

        /* Here minimum value of 13+base-vec_in[i] will be 5 */
        shift = (uint8_t)__builtin_pulp_clipu_r(14 + base - vec_in[i], 31);
        p_out[i] = (int8_t)__builtin_pulp_clip_r((output_base >> shift), 127);
    }
}
