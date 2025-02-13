#include <cuComplex.h>
#include "transpose.cuh"
#include <stdio.h>

// Déclarer les premiers &arguments avant d'appeler cette fonction
void get_args_transpose(dim3 &grid, dim3 &threads, unsigned int &shared, int block_size, int widthA, int heightA) {
  threads.x = block_size;
  threads.y = block_size;
  grid.x = std::ceil((float)widthA / (float)threads.x);
  grid.y = std::ceil((float)heightA / (float)threads.y);
  shared = block_size * (block_size+1) * sizeof(float);
}


__global__ void transpose(float* A, float* T, int heightA, int widthA) {
    int Ax = blockDim.x * blockIdx.x + threadIdx.x;
    int Ay = blockDim.y * blockIdx.y + threadIdx.y;

    int blockDimY = blockDim.y + 1;
    extern __shared__ float tile[];

    int indexTile = threadIdx.x + blockDimY * threadIdx.y;

    // Load data into shared memory, handling out-of-bounds threads
    if (Ax < widthA && Ay < heightA) {
        int indexA = Ax + widthA * Ay;
        tile[indexTile] = A[indexA];
    } else {
        tile[indexTile] = 0.0f; // Optional, as writes to T are guarded
    }

    __syncthreads(); // All threads participate

    // Calculate transposed indices and write to T if in bounds
    indexTile = threadIdx.x + blockDimY * threadIdx.y;
    int indexT = Ay + heightA * Ax;

    if (Ax < widthA && Ay < heightA) {
        T[indexT] = tile[indexTile];
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

