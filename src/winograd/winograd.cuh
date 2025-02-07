#include <stdint.h>

#ifndef WINOGRAD_CUH
#define WINOGRAD_CUH


__device__ void fetch_input_tile(float *input, float* tile, int w_input, int h_input, int tile_size);


/*
Winograd Multiplication for Convolution:
    Input: float* input matrix padded
    Output: float* output matrix padded
    filter: float* correspond to the 3x3 filter
    w_input: unsigned int w_input padded to divisible by 4
    h_input: unsigned int w_input padded to divisible by 4
*/ 
__global__ void winograd_kernel(float* output, const float* input, const float* filter, unsigned int w_input, unsigned int h_input, unsigned int w_filter, unsigned int h_filter);

void winograd_host(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter);

#endif // WINOGRAD_CUH