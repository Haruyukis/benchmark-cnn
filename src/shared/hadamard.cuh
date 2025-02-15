#include <stdint.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#ifndef HADAMARD_CUH
#define HADAMARD_CUH

/*
Element-Wise Multiplication:
    A: float*
    B: float*
    C: float* A*B
*/ 
__global__ void hadamard_kernel(float* C, const float* A, const float* B, unsigned int width, unsigned int height);

__global__ void hadamard_kernel_Cufloatc(cuFloatComplex* A, const cuFloatComplex* B, unsigned int width, unsigned int height);

void hadamard(float* C, const float* A, const float* B, int width, int height);

#endif // HADAMARD_CUH