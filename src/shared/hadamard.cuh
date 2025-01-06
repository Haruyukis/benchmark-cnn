#include <stdint.h>
#include "atoms/matrix_vo.hpp"

/*
Element-Wise Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: MatrixVo A*B
*/ 
__global__ void hadamard(MatrixVo C, const MatrixVo A, const MatrixVo B, unsigned int width, unsigned int height);