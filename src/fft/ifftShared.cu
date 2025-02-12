#include <cuda_runtime.h>
#include <cuComplex.h>
#include "fftSharedRow.cuh"
#include "../shared/transpose.cuh"
#include "./ifftShared.cuh"

/*
Inverse Fast Fourier Transform: Do the Inverse Fast Fourier Transform of imgDevice (2D FFT).
    img_complexe: image on the host ->> only for tests TO BE REMOVED
    imgDevice: image on the device
    width: width of the image
    height: height of the image
    channels: number of channels of the image
*/
void ifftShared(cuFloatComplex* imgDevice, int width, int height, int channels){
    int N = width*height;
    int log2width = (int)log2(width);
    int log2height = (int)log2(height);
    
    // For each channel, do the 2D iFFT
    for (int channel = 0; channel<channels; channel++){
        cuFloatComplex* ptrChannel = imgDevice + channel * N;
        cuFloatComplex* dataTransposed;
        cudaMalloc((void**)&dataTransposed ,N*sizeof(cuFloatComplex));
        
        // Design space for the iFFT
        dim3 threadsPerBlockFFT(width);    // One thread per element in the row
        dim3 blocksPerGridFFT(height);     // One block per row
        int sharedMemorySizeFFT = width * sizeof(cuFloatComplex); // Allocate shared memory for each row
        
        // Design space for transpose
        dim3 blockDimTranspose(32, 32);
        dim3 gridDimTranspose((width + blockDimTranspose.x - 1) / blockDimTranspose.x, (height + blockDimTranspose.y - 1) / blockDimTranspose.y);
        size_t sharedMemSizeTranspose = blockDimTranspose.x * (blockDimTranspose.y + 1) * sizeof(cuFloatComplex);
        
        // 2D iFFT
        transposeCF<<<gridDimTranspose, blockDimTranspose, sharedMemSizeTranspose>>>(ptrChannel, dataTransposed, width, height);
        cudaDeviceSynchronize();

        ifft_DIT_on_rows<<<blocksPerGridFFT, threadsPerBlockFFT, sharedMemorySizeFFT>>>(dataTransposed, width, height, log2width);
        cudaDeviceSynchronize();

        transposeCF<<<gridDimTranspose, blockDimTranspose, sharedMemSizeTranspose>>>(dataTransposed, ptrChannel, height, width);
        cudaDeviceSynchronize();
        
        ifft_DIT_on_rows<<<blocksPerGridFFT, threadsPerBlockFFT, sharedMemorySizeFFT>>>(ptrChannel, height, width, log2height);
        cudaDeviceSynchronize();
        
        cudaFree(dataTransposed);
        // cudaMemcpy(img_complexe[channel], ptrChannel, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }
}