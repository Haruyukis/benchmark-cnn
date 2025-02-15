#include "transpose.cu"

/*
Matrix Multiplication:
    A: MatrixVo
    B: MatrixVo
    C: A.B
*/ 

// Déclarer les premiers &arguments avant d'appeler cette fonction
void get_args_gemm(dim3 &grid, dim3 &threads, unsigned int &shared, int widthA, int heightA, int widthB, int block_size) {
  threads.x = block_size;
  threads.y = block_size;
  grid.x = ceil((float)widthB / (float)block_size);
  grid.y = ceil((float)heightA / (float)block_size);
  shared = 2 * widthA * block_size * sizeof(float);
}

__global__ void
gemm_kernel(float* A, float* Bt, float* C, int widthA, int heightA, int widthB) // Bt: B transposed
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
      // printf("A[%i] = %f\n", indexInitA + indexTile, A[indexInitA + indexTile]);
    }
  }

  if (indexInitBt + offsetThreadBt < nB) {
    for (int i = threadIdx.y; i < widthA; i += blockDim.y) {
      indexTile = i + offsetThreadBt;
      tileBt[indexTile] = Bt[indexInitBt + indexTile];
      // printf("Bt[%i] = %f\n", indexInitBt + indexTile, Bt[indexInitBt + indexTile]);
    }
  }

  __syncthreads();
    
  if (indexInitA + offsetThreadA < nA && indexInitBt + offsetThreadBt < nB) {
    float acc = 0.0;
    for(int k = 0; k < widthA; k++){
      acc += tileA[offsetThreadA + k] * tileBt[offsetThreadBt + k];
    }
    C[indexC] = acc;
    // printf("C[%i] = %f\n", indexC, acc);
  }
}

// Final GEMM function
// Takes arguments already on device 
void gemm(float* dA, float* dB, float* dC, int widthA, int heightA, int widthB, int block_size) {
  float* dBt;
  unsigned int mem_size_Bt = widthB * widthA * sizeof(float);
  cudaMalloc((void **)&dBt, mem_size_Bt);
  dim3 transposeGrid, transposeThreads;
  unsigned int transposeShared;
  get_args_transpose(transposeGrid, transposeThreads, transposeShared, block_size, widthB, widthA);
  transpose<<<transposeGrid, transposeThreads, transposeShared>>>(dB, dBt, widthA, widthB);
  cudaDeviceSynchronize(); 
  dim3 gemmGrid, gemmThreads;
  unsigned int gemmShared;
  get_args_gemm(gemmGrid, gemmThreads, gemmShared, widthA, heightA, widthB, block_size);
  gemm_kernel<<<gemmGrid, gemmThreads, gemmShared>>>(dA, dBt, dC, widthA, heightA, widthB);
  cudaFree(dBt);
}