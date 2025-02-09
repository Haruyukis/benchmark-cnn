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

__device__ void store_and_transform_output_tile(float* output, float *tmp, int w_output, int h_output, int tile_size, int block_x, int block_y, int thread_x, int thread_y, int nb_tiles_per_row){
    int tile_x = (thread_x + block_x * nb_tiles_per_row) << 1;
    int tile_y = (thread_y + block_y * nb_tiles_per_row) << 1;

    int tile_idx = tile_x + tile_y * w_output;

    output[tile_idx] = tmp[0] - tmp[12] - tmp[3] + tmp[15];
    output[tile_idx + 1] = - tmp[1] + tmp[13];
    output[tile_idx + w_output] = - tmp[4] + tmp[7];
    output[tile_idx + w_output + 1] = tmp[5];
}




__global__ void winograd_kernel(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    // __shared__ float transformed_filter[w_filter*h_filter]; // TODO put channel in it.
    __shared__ float transformed_input_smem[16*16][16]; // Each thread within a block will transform and store one input tile.
    
    float transformed_filter[16] = {1.0f, 0.f, 0.f, -1.0f,
        1.5f, 0.f, 0.f, -1.5f,
        0.5f, 0.f, 0.f, -0.5f,
        1.0f, 0.f, 0.f, -1.0f
    };

    int tile_size = 4;

    float input_tile[16]; // Pre-fetched input by one thread.
    int idx_smem = threadIdx.y * blockDim.x + threadIdx.x;

    fetch_input_tile(input, input_tile, w_input, h_input, tile_size, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);

    transform_input_tile(transformed_input_smem[idx_smem], input_tile);
    
    float tmp[16];
    // Hadamard Product between transformed input and transformed filter;
    for (int i = 0; i < 16; i++){
        tmp[i] = transformed_input_smem[idx_smem][i] * transformed_filter[i];
    }

    // // Put back the output.
    store_and_transform_output_tile(output, tmp, w_input, h_input, 2, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);


    // Hadamard Product
    
    // Test 
    // if (threadIdx.x == 6 && threadIdx.y == 6 && blockIdx.x == 0 && blockIdx.y == 0){
    //     fetch_input_tile(input, input_tile, w_input, h_input, tile_size, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);
    //     printf("Thread Id: (%u, %u)\n", (threadIdx.x + blockDim.x * blockIdx.x)*2, (threadIdx.y + blockDim.y * blockIdx.y)*2);
    //     for (int i = 0; i < 4; i++){
    //         for (int j=0; j < 4; j++){
    //             printf("%f ", input_tile[i * 4 + j]);
    //         }
    //         printf("\n");
    //     }

    //     transform_input_tile(transformed_input_smem[idx_smem], input_tile);
    //     printf("Thread Id of the Transformed input tile: (%u, %u)\n", (threadIdx.x + blockDim.x * blockIdx.x)*2, (threadIdx.y + blockDim.y * blockIdx.y)*2);
    //     for (int i = 0; i < 4; i++){
    //         for (int j=0; j < 4; j++){
    //             printf("%f ", transformed_input_smem[idx_smem][i * 4 + j]);
    //         }
    //         printf("\n");
    //     }
        
    //     float tmp[16];
    //     float test_output[4];
    //     // Hadamard Product between transformed input and transformed filter;
    //     for (int i = 0; i < 16; i++){
    //         tmp[i] = transformed_input_smem[idx_smem][i] * transformed_filter[i];
    //         printf("%f ", tmp[i]);
    //     }

    //     // // Put back the output.
    //     store_and_transform_output_tile(output, tmp, w_input, h_input, 2, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, blockDim.x);
    // }
}

void winograd_host(float* output, float* input, float* filter, int w_input, int h_input, int w_filter, int h_filter){
    float *d_input, *d_output;

    size_t d_input_size = w_input*h_input * sizeof(float);
    // size_t d_output_size = (w_input - w_filter + 1) * (h_input - h_filter + 1) * sizeof(float);

    cudaMalloc((void **) &d_input, d_input_size);
    cudaMalloc((void **) &d_output, d_input_size);    


    cudaError_t err = cudaMemcpy(d_input, input, d_input_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    // dim3 gridDim(ceil(((w_input - w_filter + 1 +)/2.0)/16.0), ceil(((h_input - h_filter + 1)/2.0)/16.0));
    dim3 gridDim(1, 1);

    winograd_kernel<<<gridDim, blockDim>>>(d_output, d_input, filter, w_input, h_input, w_filter, h_filter);

    cudaMemcpy(output, d_output, d_input_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);


    
}