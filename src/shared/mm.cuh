#include "atoms/tensor_vo.hpp"

/*
Matrix Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: A.B
*/ 

// Déclarer les premiers &arguments avant d'appeler cette fonction
void get_args_mm(unsigned int &mem_size_C, dim3 &grid, dim3 &threads, unsigned int &shared, int wA, int hA, int wB, int block_size) {
  mem_size_C = hA * wB * sizeof(float);
  threads.x = block_size;
  threads.y = block_size;
  grid.x = wB / block_size;
  grid.y = hA / block_size;
  shared = 2 * wA * block_size * sizeof(float);
}

__global__ void
mm(float* C, float* A, float* Bt, int widthA, int heightA, int widthB) // Bt: B transposed
{
  int nA = widthA * heightA;
  int nB = widthA * widthB;
  int xIndexInit = blockDim.x * blockIdx.x;
  int yIndexInit = blockDim.y * blockIdx.y;
  int indexInitA = widthA * yIndexInit;
  int indexInitBt = widthA * xIndexInit;
  int offsetThreadA = widthA * threadIdx.y;
  int offsetThreadBt = widthA * threadIdx.x;
  int indexC = widthB * (yIndexInit + threadIdx.y) + xIndexInit + threadIdx.x;

  extern __shared__ float tileA[];
  int middle = widthA * blockDim.y;
  float* tileBt = &tileA[middle];

  int indexTile;

  if (indexInitA + offsetThreadA < nA) {
    for (int i = threadIdx.x; i < widthA; i += blockDim.x) {
      indexTile = i + offsetThreadA;
      tileA[indexTile] = A[indexInitA + indexTile];
    }
  }

  if (indexInitBt + offsetThreadBt < nB) {
    for (int i = threadIdx.y; i < widthA; i += blockDim.y) {
      indexTile = i + offsetThreadBt;
      tileBt[indexTile] = Bt[indexInitBt + indexTile];
    }
  }

  __syncthreads();
    
  if (indexInitA + offsetThreadA < nA && indexInitBt + offsetThreadBt < nB) {
    float acc = 0.0;
    for(int k = 0; k < widthA; k++){
      acc += tileA[offsetThreadA + k] * tileBt[offsetThreadBt + k];
    }
    C[indexC] = acc;
  }
}