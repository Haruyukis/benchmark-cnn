#include "winograd.cuh"
#include <iostream>


__device__ void fetch_input_tile(float *input, float* tile, int w_input, int h_input, int tile_size, int tile_x, int tile_y, int x, int y){
    // Each thread within a block will deal with one tile (4x4 element), read in gmem.
    

    for (int i=0; i<4;i++){
        for (int j=0; j<4; j++){
            tile[i * tile_size + j] = input[tile_idx + i * w_input + j]; // TODO change to i << 2 way faster only if tile_size = 4 fixed in the future, do not read coalesced...
            if (i == 0 && j == 0){
                printf("Tile Block Idx: %u\n for the elem in input: %f\n", tile_idx, tile[i * tile_size + j]);   
            }
        }
    }
}



__global__ void winograd_kernel(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    // __shared__ float transformed_filter[w_filter*h_filter]; // TODO put channel in it.
    // __shared__ float transformed_input_smem[16][gridDim.y][gridDim.x]; // Each thread within a block will transform and store one input tile.

    int tile_size = 4;

    float input_tile[16];
    // 

    fetch_input_tile(input, input_tile, w_input, h_input, tile_size, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

    // for (int i = 0; i < 16; i++){
    //     printf("%f\n", input_tile[i]);
    // }

}

void winograd_host(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    float *d_input, *d_output;

    size_t d_input_size = w_input*h_input * sizeof(float);
    size_t d_output_size = (w_input - w_filter + 1) * (h_input - h_filter + 1) * sizeof(float);

    cudaMalloc((void **) &d_input, d_input_size);
    cudaMalloc((void **) &d_output, d_output_size);

    cudaMemcpy(d_input, input, d_input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(49, 1);
    dim3 gridDim(1, 1);

    winograd_kernel<<<gridDim, blockDim>>>(d_output, d_input, filter, w_input, h_input, w_filter, h_filter);

    cudaMemcpy(output, d_output, d_output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);


    
}