#include "data_process.cuh"
/**
 * True input to input tile arrangement
 */
__global__ void reshapeToTilesKernel(const float *d_input, float *d_input_tile, int width, int height){
    __shared__ float input_tile_smem[];

    unsigned int tile_size = 4;
    unsigned int output_tile_size = 2;

    unsigned int nb_tiles_row = (width - 2) / 2; // output_size / output_tile_size
    unsigned int nb_tiles_col = (height - 2) / 2;


}

void reshapeToTilesHost(float *input, float* input_tile, int width, int height){
    float *d_input, *d_input_tile;
    size_t size = width * height * sizeof(float);

    unsigned int tile_size = 4;
    unsigned int output_tile_size = 2;

    unsigned int output_w = width - 3 + 1;
    unsigned int output_h = height - 3 + 1;

    unsigned int nb_tiles_row = output_h / output_tile_size; // output_size / output_tile_size
    unsigned int nb_tiles_col = output_w / output_tile_size;

    cudaMalloc((void **) &d_input, size);
    cudaMalloc((void **) &d_input_tile, output_w * output_h * sizeof(float));

    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);


    dim3 blockDim(32, 32);
    dim3 gridDim(());

    cudaMemcpy(input_tile, d_input_tile, output_w * output_h * sizeof(float), cudaMemcpyDeviceToHost); // To remove at the end because we want to keep input_tiles in the device memory without copying back.

    cudaFree(d_input);
    cudaFree(d_input_tile);


}


/**
 * Output tile arrangement to true output.
 */
void rearrange_to_2x2_grid_major(float *input_image, float *output_image, int n);