#include "atoms/tensor_vo.hpp"

/*
Matrix Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: A.B
*/ 
__global__ void
mm(float* C, float* A, float* Bt, int width, int height) // Bt: B transposed
{
  int xIndexInit = blockDim.x * blockIdx.x;
  int yIndexInit = blockDim.y * blockIdx.y;
  int indexInitA = width * yIndexInit;
  int indexInitBt = width * xIndexInit;
  int offsetThreadA = width * threadIdx.y;
  int offsetThreadBt = width * threadIdx.x;
  int indexC = height * (yIndexInit + threadIdx.y) + xIndexInit + threadIdx.x;

  extern __shared__ float tileA[];
  int middle = width * blockDim.y;
  float* tileBt = &tileA[middle + 1];

  int indexTile;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    indexTile = i + offsetThreadA;
    tileA[indexTile] = A[indexInitA + indexTile];
    tileBt[indexTile] = Bt[indexInitBt + indexTile];
  }
  
  float acc = 0.0;
  
  for(int k = 0; k < width; k++){
    acc += tileA[offsetThreadA + k] * tileBt[offsetThreadBt + k];
  }

  C[indexC] = acc;
}