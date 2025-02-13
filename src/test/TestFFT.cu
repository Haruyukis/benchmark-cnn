#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "fftSharedRow.cuh"

int main(){
    int N = 12;
    cuFloatComplex* testFFT_h = (cuFloatComplex*) malloc(N*sizeof(cuFloatComplex));
    cuFloatComplex* out = (cuFloatComplex*) malloc(N*sizeof(cuFloatComplex));
    for (int i = 0; i<N; i++){
        testFFT_h[i] = make_cuFloatComplex(cosf(2.0f * M_PI * i / N), sinf(2.0f * M_PI * i / N));
        printf("Input[%d] = (%.2f, %.2f)\n", i, cuCrealf(testFFT_h[i]), cuCimagf(testFFT_h[i]));
        out[i] = make_cuFloatComplex(0,0);
    }

    cuFloatComplex* testFFT_d;
    cudaMalloc(&testFFT_d, N*sizeof(cuFloatComplex));
    cudaMemcpy(testFFT_d, testFFT_h, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    // Design space for the FFT
    dim3 threadsPerBlock(2*N);    // One thread per element in the row
    dim3 blocksPerGrid(2*1);     // One block per row
    int sharedMemorySize = N * sizeof(cuFloatComplex) * 10; // Allocate shared memory for each row

    fft_DIF_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(testFFT_d, N, 1, (int) log2(N));
    cudaDeviceSynchronize();

    cudaMemcpy(out, testFFT_d, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("\n");
    for (int i = 0; i<N; i++){
        printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(out[i]), cuCimagf(out[i]));
    }
    cudaFree(testFFT_d);
    free(testFFT_h);
    return 0;
}