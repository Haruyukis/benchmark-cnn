#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../shared/loadImage.hpp"
#include "../shared/storeImage.hpp"
#include "convFFTShared.cuh"
#include "../shared/loadImageGPU.cuh"


// Main program
int main(){
    int trueWidth, trueHeight, width, height, channels;
    const char* path = "./data/Te-noTr_0000.jpg";
    cuFloatComplex* imgDevice = loadImageGPU(path, &trueWidth, &trueHeight, &width, &height, &channels);
    cuFloatComplex** img_complexe = (cuFloatComplex**)malloc(channels*sizeof(cuFloatComplex*));
    int N = width*height;
    for (int channel = 0; channel < channels; channel++){
        img_complexe[channel] = (cuFloatComplex *)malloc(N * sizeof(cuFloatComplex));
        cudaMemcpy(img_complexe[channel], channel*N*sizeof(cuFloatComplex) + imgDevice, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }
    cuFloatComplex*kernel_h = (cuFloatComplex*)calloc(N,sizeof(cuFloatComplex));
    kernel_h[0*width + 0] = make_cuFloatComplex(-1,0);
    kernel_h[0*width + 1] = make_cuFloatComplex(-1,0);
    kernel_h[0*width + 2] = make_cuFloatComplex(-1,0);

    kernel_h[1*width + 0] = make_cuFloatComplex(0,0);
    kernel_h[1*width + 1] = make_cuFloatComplex(0,0);
    kernel_h[1*width + 2] = make_cuFloatComplex(0,0);

    kernel_h[2*width + 0] = make_cuFloatComplex(1,0);
    kernel_h[2*width + 1] = make_cuFloatComplex(1,0);
    kernel_h[2*width + 2] = make_cuFloatComplex(1,0);
    cuFloatComplex* kernel_d;
    cudaMalloc((void**)&kernel_d, N*sizeof(cuFloatComplex));
    cudaMemcpy(kernel_d, kernel_h, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

    convFFTShared(img_complexe, imgDevice, kernel_d, width, height, channels);

    // Get data from device
    for (int channel = 0; channel<channels; channel++){
        cuFloatComplex* ptrChannel = imgDevice + channel * N;   // ptr to the channel
        cudaMemcpy(img_complexe[channel], ptrChannel, N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    }

    // TAF de storeGPU
    float** image = (float**)malloc(channels*sizeof(float*));
    for (int channel = 0; channel < channels; channel++){
        image[channel] = (float*)malloc(N*sizeof(float));
    }
    // Output the result
    for (int channel = 0; channel<channels; channel++){
        for (int i = 0; i < N; ++i) {
        image[channel][i] = cuCrealf(img_complexe[channel][i]);
        // printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(h_input[i]), cuCimagf(h_input[i]));
        }
    }
    const char* chemin_sortie_inv = "./data/test 64_INVERSE?.jpeg";
    storeImageF(chemin_sortie_inv, image, width, height, channels);
    return 0;
}

/*
nvcc -ccbin /usr/bin/gcc-10 src/fft/TestFFTRows.cu src/shared/loadImage.c src/shared/storeImage.c -o build/TestFFTRows -lm -g
*/
