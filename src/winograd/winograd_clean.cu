#include "winograd.cuh"
#include <iostream>
#include <cmath>

__device__ void fetch_input_tile(float *input, float* tile, int w_input, int h_input, int tile_size, int block_x, int block_y, int thread_x, int thread_y, int nb_tiles_per_row){
    // Each thread within a block will deal with one tile (4x4 element), read in gmem.
    int tile_x = (thread_x + block_x * nb_tiles_per_row) << 1;
    int tile_y = (thread_y + block_y * nb_tiles_per_row) << 1;

    int tile_idx = tile_x + tile_y * w_input;


    if (tile_x < w_input - 2 && tile_y < h_input - 2){
        for (int i=0; i < 4;i++){
            for (int j=0; j < 4; j++){
                tile[i * tile_size + j] = input[tile_idx + i * w_input + j]; // TODO change to i << 2 way faster only if tile_size = 4 fixed in the future, do not read coalesced...
            }
        }
    } else {
        for (int i=0; i<4; i++){
            for (int j = 0; j<4; j++){
                tile[i * tile_size + j] = 0;
            }
        }
    }
}



__global__ void winograd_kernel(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    // __shared__ float transformed_filter[w_filter*h_filter]; // TODO put channel in it.
    // __shared__ float transformed_input_smem[16][gridDim.y][gridDim.x]; // Each thread within a block will transform and store one input tile.

    int tile_size = 4;

    float input_tile[16];
    fetch_input_tile(input, input_tile, w_input, h_input, tile_size, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);
    
}

void winograd_host(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    float *d_input, *d_output;

    size_t d_input_size = w_input*h_input * sizeof(float);
    size_t d_output_size = (w_input - w_filter + 1) * (h_input - h_filter + 1) * sizeof(float);

    cudaMalloc((void **) &d_input, d_input_size);
    cudaMalloc((void **) &d_output, d_output_size);    


    cudaError_t err = cudaMemcpy(d_input, input, d_input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    // dim3 gridDim(ceil(((w_input - w_filter + 1 +)/2.0)/16.0), ceil(((h_input - h_filter + 1)/2.0)/16.0));
    dim3 gridDim(1, 1);

    winograd_kernel<<<gridDim, blockDim>>>(d_output, d_input, filter, w_input, h_input, w_filter, h_filter);

    cudaMemcpy(output, d_output, d_output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);


    
}