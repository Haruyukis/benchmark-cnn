#include <stdint.h>

#ifndef WINOGRAD_CUH
#define WINOGRAD_CUH

/*
Element-Wise Multiplication:
    A: float*
    B: float*
    C: float* A*B
*/ 
__global__ void winograd_kernel(float* C, const float* A, const float* B, unsigned int width, unsigned int height);

void winogradHost(float* C, const float* A, const float* B, int width, int height);

#endif // WINOGRAD_CUH