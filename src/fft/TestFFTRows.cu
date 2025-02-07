#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../shared/loadImage.hpp"
#include "../shared/storeImage.hpp"
#include "../shared/hadamard.cuh"
#include "./fftShared.cuh"
#include "./ifftShared.cuh"
#include "../shared/transpose.cuh"


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
    return 0;
}

/*
nvcc -ccbin /usr/bin/gcc-10 src/fft/TestFFTRows.cu src/shared/loadImage.c src/shared/storeImage.c -o build/TestFFTRows -lm -g
*/
