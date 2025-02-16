#include "winograd/winograd.cuh"
#include "shared/loadImageGPU.cuh"
#include "shared/storeImageGPU.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include "shared/utils.cpp"
#include "shared/storeImage.hpp"

cudaEvent_t start, stop;

void startTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
}

void stopTimer(const char* label) {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << label << " took " << milliseconds << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, "Usage: %s <chemin_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* path = argv[1];
    int width, height, nb_channels;
    startTimer(); // Start the timer before loading the image
    float* d_input = loadImageGPUf(path, &width, &height, &nb_channels);
    stopTimer("loadImageGPUf"); // Stop the timer and print the time taken

    float* d_output;
    
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* filter = new float[9]{
        -1.0f, -1.f, -1.f,
        0.0f, 0.f, 0.f,
        1.0f, 1.f, 1.f
    };
    
    cudaMalloc((void **) &d_output, o_size * sizeof(float) * nb_channels);
    startTimer(); // Start timer before calling winograd_host
    winograd_host(d_output, d_input, filter, width, height, 3, 3, nb_channels, 1);
    stopTimer("winograd_host"); // Stop timer and print time

    startTimer(); // Start timer before saving the image
    storeImageGPUf(d_output, "output_gpu.jpg", o_width, o_height, nb_channels);
    stopTimer("storeImageGPUf"); // Stop timer and print time

    
    return 0;
}