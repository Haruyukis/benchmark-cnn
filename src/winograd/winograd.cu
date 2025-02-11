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
                tile[(i << 2) + j] = input[tile_idx + i * w_input + j]; // TODO change to i << 2 way faster only if tile_size = 4 fixed in the future, do not read coalesced...
            }
        }
    } else {
        for (int i=0; i<4; i++){
            for (int j = 0; j<4; j++){
                tile[(i << 2) + j] = 0;
            }
        }
    }
}

__device__ void transform_input_tile(float *transformed_input_tile, float *input_tile){

    float workspace[4];
    // In-place computation B^T*input_tile
    for(int j=0; j<4; j++){
      workspace[0] = input_tile[j];
      workspace[1] = input_tile[j+4];
      workspace[2] = input_tile[j+8];
      workspace[3] = input_tile[j+12];
  
      input_tile[j]    = workspace[0] - workspace[2];
      input_tile[j+4]  = workspace[1] + workspace[2];
      input_tile[j+8]  = workspace[2] - workspace[1];
      input_tile[j+12] = workspace[1] - workspace[3];
    }

    // ((B^T*input_tile) * B)^T => but then we put smartly directly to the right index (no transpose) // In-place matrix mutliplication.
    for (int i = 0; i < 4; i++){
        transformed_input_tile[(i << 2)] = input_tile[(i << 2)] - input_tile[(i << 2) + 2];
        transformed_input_tile[(i << 2) + 1] = input_tile[(i << 2) + 1] + input_tile[(i << 2) + 2];
        transformed_input_tile[(i << 2) + 2] = input_tile[(i << 2) + 2] - input_tile[(i << 2) + 1];
        transformed_input_tile[(i << 2) + 3] = input_tile[(i << 2) + 1] - input_tile[(i << 2) + 3]; 
    }
}

__device__ void store_and_transform_output_tile(float* output, float *tmp, int w_output, int h_output, int block_x, int block_y, int thread_x, int thread_y, int nb_tiles_per_row){
    int tile_x = (thread_x + block_x * nb_tiles_per_row) << 1;
    int tile_y = (thread_y + block_y * nb_tiles_per_row) << 1;

    int tile_idx = tile_x + tile_y * w_output;
    
    // TODO reduce redondant computation for the sum.
    if (tile_x < w_output && tile_y < h_output){
        output[tile_idx] = (tmp[0] + tmp[4] + tmp[8]) + (tmp[1] + tmp[5] + tmp[9]) + (tmp[2] + tmp[6] + tmp[10]);
        output[tile_idx + 1] = (tmp[1] + tmp[5] + tmp[9]) - (tmp[2] + tmp[6] + tmp[10]) - (tmp[3] + tmp[7] + tmp[11]);
        output[tile_idx + w_output] = (tmp[4] - tmp[8] - tmp[12]) + (tmp[5] - tmp[9] - tmp[13]) + (tmp[6] - tmp[10] - tmp[14]);
        output[tile_idx + w_output + 1] = (tmp[5] + tmp[9] + tmp[13]) - (tmp[6] + tmp[10] + tmp[14]) - (tmp[7] + tmp[11] + tmp[15]);
    }
    // output[tile_idx] = (tmp[0] + tmp[4] + tmp[8]) + (tmp[1] + tmp[5] + tmp[9]) + (tmp[2] + tmp[6] + tmp[10]);
    // output[tile_idx + 1] = (tmp[1] + tmp[5] + tmp[9]) - (tmp[2] + tmp[6] + tmp[10]) - (tmp[3] + tmp[7] + tmp[11]);
    // output[tile_idx + w_output] = (tmp[4] - tmp[8] - tmp[12]) + (tmp[5] - tmp[9] - tmp[13]) + (tmp[6] - tmp[10] - tmp[14]);
    // output[tile_idx + w_output + 1] = (tmp[5] + tmp[9] + tmp[13]) - (tmp[6] + tmp[10] + tmp[14]) - (tmp[7] + tmp[11] + tmp[15]);
    // if (threadIdx.x == 0 && threadIdx.y == 1){
    //     printf("%f %f %f %f", output[tile_idx], output[tile_idx + 1], output[tile_idx + w_output], output[tile_idx + w_output + 1]);
    //     printf("\n");
    // }        
}

__global__ void winograd_kernel(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter, int w_output, int h_output){
    // __shared__ float transformed_filter[w_filter*h_filter]; // TODO put channel in it.
    __shared__ float transformed_input_smem[16*16][16]; // Each thread within a block will transform and store one input tile.
    
    float transformed_filter[16] = {1.0f, 0.f, 0.f, -1.0f,
        1.5f, 0.f, 0.f, -1.5f,
        0.5f, 0.f, 0.f, -0.5f,
        1.0f, 0.f, 0.f, -1.0f
    };

    int tile_size = 4;

    float input_tile[16]; // Pre-fetched input by one thread.
    float accumulator[16]; // Workspace for Hadamard product.
    int idx_smem = threadIdx.y * blockDim.x + threadIdx.x;

    fetch_input_tile(input, input_tile, w_input, h_input, tile_size, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);
    __syncthreads();
    // ITF
    transform_input_tile(transformed_input_smem[idx_smem], input_tile);
    __syncthreads();


    // Hadamard Product
    for (int i=0; i < 16; i++){
        accumulator[i] = transformed_input_smem[idx_smem][i] * transformed_filter[i];
    }
    __syncthreads();

    store_and_transform_output_tile(output, accumulator, w_output, h_output, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);
}

void winograd_host(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    float *d_input, *d_output;

    size_t d_input_size = w_input*h_input * sizeof(float);
    int w_output = (w_input - w_filter + 1);
    int h_output = (h_input - h_filter + 1);
    size_t d_output_size = w_output * h_output * sizeof(float);

    cudaMalloc((void **) &d_input, d_input_size);
    cudaMalloc((void **) &d_output, d_output_size);    


    cudaError_t err = cudaMemcpy(d_input, input, d_input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    // dim3 gridDim(ceil(((w_input - w_filter + 1 +)/2.0)/16.0), ceil(((h_input - h_filter + 1)/2.0)/16.0));
    dim3 gridDim(1, 1);

    winograd_kernel<<<gridDim, blockDim>>>(d_output, d_input, filter, w_input, h_input, w_filter, h_filter, w_output, h_output);
    cudaMemcpy(output, d_output, d_output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}