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