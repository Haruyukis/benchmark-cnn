#include "winograd/winograd.cuh"
#include "shared/loadImageGPU.cuh"
#include "shared/storeImageGPU.cuh"
#include <iostream>
#include <cuda_runtime.h>

int main(int argc, char *argv[]){
    if (argc != 2){
        fprintf(stderr, "Usage: %s <chemin_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* path = argv[1];
    int width, height, nb_channels;
    float* d_input = loadImageGPUf(path, &width, &height, &nb_channels);
    float* d_output;
    
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    // float* input = new float[i_size];
    // float* output = new float[o_size * nb_channels];
    float* filter = new float[9]{
        -1.0f, -1.f, -1.f,
        0.0f, 0.f, 0.f,
        1.0f, 1.f, 1.f
    };
    
    cudaMalloc((void **) &d_output, o_size * sizeof(float) * nb_channels);
    winograd_host(d_output, d_input, filter, width, height, 3, 3, nb_channels, 1);


    // cudaMemcpy(d_output, output, o_size * sizeof(float) * nb_channels, cudaMemcpyHostToDevice);

    storeImageGPUf(d_output, "output.jpg", o_width, o_height, nb_channels);
    
    return 0;
}