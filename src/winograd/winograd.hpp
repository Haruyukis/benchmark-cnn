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
void transform_input(float *transformed_input_tile, float* input_tile, float* B, float* B_t);

void transform_filter(float *transformed_filter, float* filter, float* G, float *G_t);

float* winograd_cpu(float* output, float* input, float* filter, unsigned int w_input, unsigned int h_input, unsigned int w_filter, unsigned int h_filter);



#endif // WINOGRAD_HPP