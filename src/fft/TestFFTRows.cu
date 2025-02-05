#include "fft_shared_row.cu"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "../shared/transpose.cuh"
#include "../shared/loadImage.hpp"
#include "../shared/storeImage.hpp"

void fftShared(cuFloatComplex** img_complexe, int width, int height, int channels){
    int N = width*height;
    int log2width = (int)log2(width);
    int log2height = (int)log2(height);
    // For each channel, do the FFT
    for (int channel = 0; channel<channels; channel++){
        cuFloatComplex *d_data;
        cudaMalloc((void **)&d_data, N * sizeof(cuFloatComplex));
        // Copy input data from host to device
        cudaMemcpy(d_data, img_complexe[channel], N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(width);    // One thread per element in the row
        dim3 blocksPerGrid(height);     // One block per row
        // Allocate shared memory for each row
        int sharedMemorySize = width * sizeof(cuFloatComplex);
        // Launch FFT kernel for each row of the image
        fft_DIF_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(d_data, width, height, log2width);

        // Wait for kernel to finish
        cudaDeviceSynchronize();
        dim3 blockDim(32, 32);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Taille de la mémoire partagée
        size_t sharedMemSize = blockDim.x * (blockDim.y + 1) * sizeof(cuFloatComplex);

        cuFloatComplex* dataTransposed;
        cudaMalloc((void**)&dataTransposed ,N*sizeof(cuFloatComplex));

        transposeCF<<<gridDim, blockDim, sharedMemSize>>>(d_data, dataTransposed, width, height);
        cudaDeviceSynchronize();

        
        fft_DIF_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(dataTransposed, height, width, log2height);
        cudaDeviceSynchronize();
        
        transposeCF<<<gridDim, blockDim, sharedMemSize>>>(dataTransposed, d_data, height, width);
        cudaDeviceSynchronize();
        cudaFree(dataTransposed);
        // Copy the result back from device to host
        
        cudaMemcpy(img_complexe[channel], d_data, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
}

// Main program
int main(){
    int width, height, channels;
    const char* chemin_image = "./data/Te-noTr_0000_padded.jpg";
    float** image = loadImageF(chemin_image, &width, &height, &channels);
    
    const int N = width * height; // Total number of elements (pixels in the image)
    const int log2width = (int)log2f(width);
    const int log2height = (int)log2f(height);

    // Fait le taf de la fonction load_complexe
    cuFloatComplex** img_complexe = (cuFloatComplex**)malloc(channels*sizeof(cuFloatComplex*));
    for (int channel = 0; channel < channels; channel++){
        img_complexe[channel] = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
        for (int n = 0; n < N; n++){
            img_complexe[channel][n] = make_cuFloatComplex(image[channel][n], 0);
        }
    }

    fftShared(img_complexe, width, height, channels);

    // Output the result
    for (int channel = 0; channel<channels; channel++){
        for (int i = 0; i < N; ++i) {
        image[channel][i] = cuCrealf(img_complexe[channel][i]);
        // printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(h_input[i]), cuCimagf(h_input[i]));
        }
    }
    const char* chemin_sortie = "./data/test fft.jpeg";
    storeImageF(chemin_sortie, image, width, height, channels);

    


    

    

    

    
    

    /*
    // REVERSE

    ifft_DIT_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(d_data, width, height, log2width);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    cudaMemcpy(h_input, d_data, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // Output the result
    for (int i = 0; i < N; ++i) {
        image[0][i] = cuCrealf(h_input[i])/width;
        // printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(h_input[i]), cuCimagf(h_input[i]));
    }
    const char* chemin_sortie_ifft = "./data/test ifft.jpeg";
    storeImageF(chemin_sortie_ifft, image, width, height, channels);

    // Clean up memory
    cudaFree(d_data);
    free(h_input);
    */

    return 0;
}

/*
nvcc -ccbin /usr/bin/gcc-10 src/fft/TestFFTRows.cu src/shared/loadImage.c src/shared/storeImage.c -o build/TestFFTRows -lm -g
*/
