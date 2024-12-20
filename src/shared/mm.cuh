#include "atoms/tensor_vo.hpp"

/*
Matrix Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: A.B
*/ 
__global__  void mm(MatrixVo C, MatrixVo A, MatrixVo B, unsigned int wA, unsigned int hA, unsigned int wA, unsigned int hB, uint8_t depth); //TODO pas sûr de la signature.