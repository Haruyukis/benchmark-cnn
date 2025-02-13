#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH
#include <stdio.h>

__global__ void
transpose(float* A, float* T, int heightA, int widthA); // A: source matrix; T: matrix to be transposed

__global__ void transposeCF(cuFloatComplex* A, cuFloatComplex* T, int widthA, int heightA); 


#endif //TRANSPOSE_CUH