# Arm Cortex-M

## CMSIS-NN
For better results consider updating "arm_softmax_q7" of CMSIS-NN.
#### Original
```c
for (i = 0; i < dim_vec; i++)
{
    /* Here minimum value of 13+base-vec_in[i] will be 5 */
    shift = (uint8_t)__USAT(13 + base - vec_in[i], 5);
    p_out[i] = (q7_t)__SSAT((output_base >> shift), 8);
}
```

#### Tuned
```c
for (i = 0; i < dim_vec; i++)
{
    /* Here minimum value of 14+base-vec_in[i] will be 5 */
    shift = (uint8_t)__USAT(14 + base - vec_in[i], 5);
    p_out[i] = (q7_t)__SSAT((output_base >> shift), 8);
}
```