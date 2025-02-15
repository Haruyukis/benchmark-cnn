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
    int trueWidth, trueHeight, width, height, channels;
    float* d_input = loadImageGPUf(path, &trueWidth, &trueHeight, &width, &height, &channels);

    int size = width * height * channels * sizeof(float);
    
    // Allouer de la mémoire sur l'hôte
    float* h_input = (float*)malloc(size);

    // Copier les données du GPU vers le CPU
    cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);

    // Afficher quelques valeurs (par exemple, les 10 premières)
    std::cout << channels << std::endl;
    std::cout << height << std::endl;
    std::cout << width << std::endl;
    std::cout << "INPUT DE POMPIDOU" << std::endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++){

            std::cout << h_input[i * width + j] << " ";
        }
        std::cout << std::endl;
    }


    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    // float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        -1.0f, -1.f, -1.f,
        0.0f, 0.f, 0.f,
        1.0f, 1.f, 1.f
    };

    winograd_host(output, h_input, filter, width, height, 3, 3, 1);
    std::cout << "OUTPUT DE POMPIDOU" << std::endl;
    for (int i=0; i<o_height; i++){
        for (int j=0; j<o_width; j++){
            std::cout << output[i * o_width + j] << " ";
        }
        std::cout << std::endl;

    }






    // Libérer la mémoire allouée sur l'hôte
    // free(h_input);

    // std::cout << trueWidth << std::endl;
    // std::cout << channels << std::endl;

    // int w_filter = 3;
    // int h_filter = 3;
    // int w_output = (trueWidth - w_filter + 1);
    // int h_output = (trueHeight - h_filter + 1);
    // int size_input = trueWidth * trueHeight;
    // int size_output = w_output * h_output;

    // // int w_output = (width - w_filter + 1);
    // // int h_output = (height - h_filter + 1);
    // // int size_input = width * height;
    // // int size_output = w_output * h_output;


    // float* output = new float[size_output];
    // float* filter = new float[9]{
    //     -1.0f, -1.0f, -1.0f,
    //     0.0f, 0.0f, 0.0f,
    //     1.0f, 1.0f, 1.0f
    // };

    // int blockSize_x = 32;
    // int blockSize_y = 32;
    // float *d_output, *d_filter;


    // // size_t d_input_size = width*height * sizeof(float) * channels;
    // size_t d_output_size = w_output * h_output * sizeof(float) * channels;
    // size_t d_filter_size = w_filter*h_filter*sizeof(float);

    // cudaMalloc((void **) &d_output, d_output_size);    
    // cudaMalloc((void **) &d_filter, d_filter_size);    

    // cudaMemcpy(d_filter, filter, d_filter_size, cudaMemcpyHostToDevice);

    // int o_offset = w_output * h_output;
    // int i_offset = width * height;

    // dim3 blockDim(blockSize_x, blockSize_y);
    // dim3 gridDim((w_output + (blockSize_x * 2 - 1)) / (blockSize_x * 2), (h_output + (blockSize_y * 2 - 1)) / (blockSize_y * 2));
    // for (int c=0; c < channels; c++){
    //     winograd_kernel<<<gridDim, blockDim>>>((d_output + c*o_offset), (d_input + c*i_offset), d_filter, trueWidth, trueHeight, w_filter, h_filter, w_output, h_output);
    // }
    // // cudaMemcpy(output, d_output, d_output_size, cudaMemcpyDeviceToHost);

    // std::cout  << size_output * channels << std::endl;

    // const char* PAF = "output.jpeg";
    
    // storeImageGPUf(d_output, PAF, w_output, h_output, channels);











    // cudaFree(d_output);
    // cudaFree(d_input);
    // cudaFree(d_filter);
    // delete[] output, filter;
    
    return 0;
}