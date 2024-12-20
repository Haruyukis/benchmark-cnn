#include <stdint.h>
#include "atoms/matrix_vo.hpp"

/*
Element-Wise Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: MatrixVo A*B
*/ 
__global__ void hadamard(MatrixVo C, MatrixVo A, MatrixVo B, unsigned int wA, unsigned int hA, unsigned int wA, unsigned int hB, uint8_t depth);