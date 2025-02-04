#include "winograd.cuh"

__global__ void winograd_kernel(float* output, const float* input, const float* filter, unsigned int w_input, unsigned int h_input, unsigned int w_filter, unsigned int h_filter){
    __shared__ float filter_smem[w_filter*h_filter];
    __shared__ float input_smem[];


}