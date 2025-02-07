#include <cuComplex.h>
#include "transpose.cuh"
#include <stdio.h>

__global__ void
transpose(float* A, float* T, int heightA, int widthA) // A: source matrix; T: matrix to be transposed
{
  int Ax = blockDim.x * blockIdx.x + threadIdx.x;
  int Ay = blockDim.y * blockIdx.y + threadIdx.y;

  if (Ax < widthA && Ay < heightA) {

    int indexA = Ax + widthA * Ay;

    int blockDimY = blockDim.y + 1;
    extern __shared__ float tile[]; // blockDimY = blockDim.y + 1 to avoid bank conflicts

    int indexTile = threadIdx.y + blockDimY * threadIdx.x;

    tile[indexTile] = A[indexA]; // Transposition

    __syncthreads();

    indexTile = threadIdx.x + blockDimY * threadIdx.y;
    int indexT = Ay + heightA * Ax;
    T[indexT] = tile[indexTile]; // Write to destination matrix
  }
  else {
    printf("Index x or y out of bounds: x = %i, y = %i\n", Ax, Ay);
  }
}

/* MÊME FONCTION QUE CELLE DU DESSUS MAIS POUR LES CuFloatComplex
/!\ ordre width-height inversé dans la signature par rapport à celle réelle
// A: source matrix; T: matrix to be transposed*/
__global__ void
transposeCF(cuFloatComplex* A, cuFloatComplex* T, int widthA, int heightA) 
{
  int Ax = blockDim.x * blockIdx.x + threadIdx.x;
  int Ay = blockDim.y * blockIdx.y + threadIdx.y;

  if (Ax < widthA && Ay < heightA) {

    int indexA = Ax + widthA * Ay;

    int blockDimY = blockDim.y + 1;
    extern __shared__ cuFloatComplex tileCF[]; // blockDimY = blockDim.y + 1 to avoid bank conflicts

    int indexTile = threadIdx.x + blockDimY * threadIdx.y;

    tileCF[indexTile] = A[indexA]; // Transposition

    __syncthreads();

    indexTile = threadIdx.x + blockDimY * threadIdx.y;
    int indexT = Ay + heightA * Ax;
    T[indexT] = tileCF[indexTile]; // Write to destination matrix
  }
  else {
    printf("Index x or y out of bounds: x = %i, y = %i\n", Ax, Ay);
  }
}