#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "../shared/loadImage.hpp"
#include "../shared/storeImage.hpp"
#include "convFFTShared.cuh"
#include "../shared/loadImageGPU.cuh"
#include "../shared/storeImageGPU.cuh"

// Main program
int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, "Usage: %s <chemin_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* path = argv[1];
    int trueWidth, trueHeight, width, height, channels;
    cuFloatComplex* imgDevice = loadImageGPU(path, &trueWidth, &trueHeight, &width, &height, &channels);
    int N = width*height;

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

    convFFTShared(imgDevice, kernel_d, width, height, channels);

    const char* PAF = "output.jpeg";
    
    storeImageGPU(imgDevice, PAF, trueWidth, trueHeight, width, height, channels);

    // Clean :
    cudaFree(imgDevice);
    free(kernel_h);
    cudaFree(kernel_d);

    return 0;
}

/*
nvcc -ccbin /usr/bin/gcc-10 src/fft/TestFFTRows.cu src/shared/loadImage.c src/shared/storeImage.c -o build/TestFFTRows -lm -g
*/
