#include <cuda_runtime.h>
#include <cuComplex.h>
#include "fftSharedRow.cuh"
#include "../shared/transpose.cuh"
#include "fftShared.cuh"

/*
Fast Fourier Transform: Do the Fast Fourier Transform of imgDevice (2D FFT).
    img_complexe: image on the host ->> only for tests TO BE REMOVED
    imgDevice: image on the device
    width: width of the image
    height: height of the image
    channels: number of channels of the image
*/
void fftShared(cuFloatComplex** img_complexe, cuFloatComplex* imgDevice, int width, int height, int channels){
    int N = width*height;
    int log2width = (int)log2(width);
    int log2height = (int)log2(height);
    
    // For each channel, do the 2D FFT
    for (int channel = 0; channel<channels; channel++){
        cuFloatComplex* ptrChannel = imgDevice + channel * N;   // ptr to the channel
        cuFloatComplex* dataTransposed; // for intermediate transpose
        cudaMalloc((void**)&dataTransposed ,N*sizeof(cuFloatComplex));

        // Design space for the FFT
        dim3 threadsPerBlock(width);    // One thread per element in the row
        dim3 blocksPerGrid(height);     // One block per row
        int sharedMemorySize = width * sizeof(cuFloatComplex); // Allocate shared memory for each row
        
        // 2D FFT
        fft_DIF_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(ptrChannel, width, height, log2width);
        cudaDeviceSynchronize();

        // Design space for transpose
        dim3 blockDimTranspose(32, 32);
        dim3 gridDimTranspose((width + blockDimTranspose.x - 1) / blockDimTranspose.x, (height + blockDimTranspose.y - 1) / blockDimTranspose.y);
        int sharedMemSizeTranspose = blockDimTranspose.x * (blockDimTranspose.y + 1) * sizeof(cuFloatComplex);
        transposeCF<<<gridDimTranspose, blockDimTranspose, sharedMemSizeTranspose>>>(ptrChannel, dataTransposed, width, height);
        cudaDeviceSynchronize();

        fft_DIF_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(dataTransposed, height, width, log2height);
        cudaDeviceSynchronize();
        
        transposeCF<<<gridDimTranspose, blockDimTranspose, sharedMemSizeTranspose>>>(dataTransposed, ptrChannel, height, width);
        cudaDeviceSynchronize();
        
        cudaFree(dataTransposed);
        cudaMemcpy(img_complexe[channel], ptrChannel, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }
}