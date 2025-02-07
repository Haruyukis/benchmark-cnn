#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../shared/loadImage.hpp"
#include "../shared/storeImage.hpp"
#include "../shared/hadamard.cu"
// #include "./fftShared.cuh"
// #include "./ifftShared.cuh"
#include "../shared/transpose.cuh"
#include "fft_shared_row.cu"

void ifftShared(cuFloatComplex** img_complexe, cuFloatComplex* imgDevice, int width, int height, int channels){
    int N = width*height;
    int log2width = (int)log2(width);
    int log2height = (int)log2(height);
    // For each channel, do the FFT
    for (int channel = 0; channel<channels; channel++){
        cuFloatComplex* ptrChannel = imgDevice + channel * N;
        dim3 threadsPerBlock(width);    // One thread per element in the row
        dim3 blocksPerGrid(height);     // One block per row
        // Allocate shared memory for each row
        int sharedMemorySize = width * sizeof(cuFloatComplex);
        // Launch FFT kernel for each row of the image
        dim3 blockDim(32, 32);
        dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

        // Taille de la mémoire partagée
        size_t sharedMemSize = blockDim.x * (blockDim.y + 1) * sizeof(cuFloatComplex);
        cuFloatComplex* dataTransposed;
        cudaMalloc((void**)&dataTransposed ,N*sizeof(cuFloatComplex));
        transposeCF<<<gridDim, blockDim, sharedMemSize>>>(ptrChannel, dataTransposed, width, height);
        cudaDeviceSynchronize();


        ifft_DIT_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(dataTransposed, width, height, log2width);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        transposeCF<<<gridDim, blockDim, sharedMemSize>>>(dataTransposed, ptrChannel, height, width);
        cudaDeviceSynchronize();

        

        
        ifft_DIT_on_rows<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(ptrChannel, height, width, log2height);
        cudaDeviceSynchronize();
        
        cudaFree(dataTransposed);
        // Copy the result back from device to host
        
        cudaMemcpy(img_complexe[channel], ptrChannel, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }
}
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

// Main program
int main(){
    int width, height, channels;
    const char* chemin_image = "./data/gris_padded.jpg";
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
    cuFloatComplex* imgDevice;
    cudaMalloc(&imgDevice, channels * N * sizeof(cuFloatComplex));

    // Step 2: Allocate memory for each channel on the device
    for (int channel = 0; channel < channels; channel++) {
    cuFloatComplex* ptrChannel = imgDevice + channel * N;  // Correct pointer arithmetic

    // Assuming img_complexe[channel * N] is the start of the channel data on the host
    cudaMemcpy(ptrChannel, img_complexe[channel], N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    }

    fftShared(img_complexe, imgDevice, width, height, channels);
    
    // Output the result
    for (int channel = 0; channel<channels; channel++){
        for (int i = 0; i < N; ++i) {
        image[channel][i] = cuCrealf(img_complexe[channel][i]);
        // printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(h_input[i]), cuCimagf(h_input[i]));
        }
    }
    const char* chemin_sortie = "./data/test fft_apres_vla_modifs.jpeg";
    storeImageF(chemin_sortie, image, width, height, channels);

    cuFloatComplex*kernel_h = (cuFloatComplex*)calloc(N,sizeof(cuFloatComplex));
    // kernel_h[127*width + 127] = make_cuFloatComplex(-1,0);
    // kernel_h[127*width + 128] = make_cuFloatComplex(1,0);
    // kernel_h[128*width + 127] = make_cuFloatComplex(-1,0);
    // kernel_h[128*width + 128] = make_cuFloatComplex(1,0);
    kernel_h[0*width + 0] = make_cuFloatComplex(0,0);
    kernel_h[0*width + 1] = make_cuFloatComplex(1,0);
    kernel_h[0*width + 2] = make_cuFloatComplex(0,0);
    kernel_h[1*width + 0] = make_cuFloatComplex(1,0);
    kernel_h[1*width + 1] = make_cuFloatComplex(-4,0);
    kernel_h[1*width + 2] = make_cuFloatComplex(1,0);
    kernel_h[2*width + 0] = make_cuFloatComplex(0,0);
    kernel_h[2*width + 1] = make_cuFloatComplex(1,0);
    kernel_h[2*width + 2] = make_cuFloatComplex(0,0);
    cuFloatComplex* kernel_d;
    cudaMalloc((void**)&kernel_d, N*sizeof(cuFloatComplex));
    cudaMemcpy(kernel_d, kernel_h, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    fftShared(img_complexe, kernel_d, width, height, 1);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Taille de la mémoire partagée
    size_t sharedMemSize = blockDim.x * (blockDim.y + 1) * sizeof(cuFloatComplex);
    for (int channel = 0; channel < channels; channel++){
        cuFloatComplex* ptrChannel = imgDevice + channel * N;
        hadamard_kernel_Cufloatc<<<gridDim, blockDim, sharedMemSize>>>(ptrChannel, kernel_d, width, height);
    }
    
    // inverse
    ifftShared(img_complexe, imgDevice, width, height, channels);
    
    // Output the result
    for (int channel = 0; channel<channels; channel++){
        for (int i = 0; i < N; ++i) {
        image[channel][i] = cuCrealf(img_complexe[channel][i]);
        // printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(h_input[i]), cuCimagf(h_input[i]));
        }
    }
    const char* chemin_sortie_inv = "./data/test fft_apres_vla_modifs_INVERSE?.jpeg";
    storeImageF(chemin_sortie_inv, image, width, height, channels);
    

    


    

    

    

    
    

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
