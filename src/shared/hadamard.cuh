#include <stdint.h>
#include "atoms/matrix_vo.hpp"

#ifndef HADAMARD_CUH
#define HADAMARD_CUH

/*
Element-Wise Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: MatrixVo A*B
*/ 
__global__ void hadamard_kernel(MatrixVo C, const MatrixVo A, const MatrixVo B, unsigned int width, unsigned int height);

void hadamard(MatrixVo C, const MatrixVo A, const MatrixVo B, int width, int height);

#endif // HADAMARD_CUH