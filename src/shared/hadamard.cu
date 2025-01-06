#include "hadamard.cuh"

__global__ void hadamard_kernel(MatrixVo C, const MatrixVo A, const MatrixVo B, unsigned int width, unsigned int height) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int index = row * width + col;

    if (row < height && col < width) {
        C[index] = A[index] * B[index];
    }
}

void hadamard(double* C, const double* A, const double* B, int width, int height) {
    double *d_A, *d_B, *d_C;
    size_t size = width * height * sizeof(double);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    
    hadamard_kernel<<<gridDim, blockDim>>>(d_C, d_A, d_B, width, height);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}