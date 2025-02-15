#include <stdint.h>
#include "../shared/mm.hpp"

#ifndef WINOGRAD_HPP
#define WINOGRAD_HPP

/*
Winograd Multiplication for Convolution:
    Input: float* input matrix padded
    Output: float* output matrix padded
    filter: float* correspond to the 3x3 filter
    w_input: unsigned int w_input padded to divisible by 4
    h_input: unsigned int w_input padded to divisible by 4
*/
void transform_input(float *transformed_input_tile, float *input_tile, float *B, float *B_t);

void transform_filter(float *transformed_filter, float *filter, float *G, float *G_t);

void transform_output(float *output, float *transformed_tile, float *transformed_filter, float *A, float *A_t);

// float *winograd_cpu(float *input, float *filter, unsigned int w_input, unsigned int h_input, unsigned int w_filter, unsigned int h_filter);

void extract_tile(float *input, float *input_tile, int startRow, int startCol, int w_input, int h_input, int w_tile, int h_tile);

void winograd_cpu(float *output, float *input, float *filter, int w_input, int h_input, int w_filter, int h_filter, int nb_channel);

#endif // WINOGRAD_HPP