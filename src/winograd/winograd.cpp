#include "winograd.hpp"
#include <iostream>
/**
 * Transforming the input_tile into the transformed one (4x4 Matrix).
 * transformed_tile: contains the operation B*input_tile*B_t.
 * input_tile: the tile that is transformed (4x4)
 */
void transform_input(float* transformed_tile, float* input_tile, float* B, float* B_t) {
    float* tmp = new float[4 * 4];
    gemm_cpu_noblas_par<float>(tmp, input_tile, B_t, 4, 4, 4);
    gemm_cpu_noblas_par<float>(transformed_tile, B, tmp, 4, 4, 4);
    delete tmp;
};

/**
 * Transform the filter for Winograd (4x4 Matrix).
 * transformed_filter: contains the operation G*filter*G_t.
 * filter: filter that is transformed (3x3 Filter)
 */

 // TODO Give tmp as a parameter to avoid copy ?
void transform_filter(float* transformed_filter, float* filter, float* G, float* G_t) {
    float* tmp = new float[3 * 4];
    gemm_cpu_noblas_par<float>(tmp, filter, G_t, 3, 4, 3);
    gemm_cpu_noblas_par<float>(transformed_filter, G, tmp, 4, 4, 3);
    delete tmp;
}

void transform_output(float* output, float* transformed_tile, float* transformed_filter, float* A, float* A_t) {
    float* tmp = new float[4 * 4];
    // Hadamard Product
    for (int i = 0; i < 16; i++) {
        transformed_tile[i] = transformed_tile[i] * transformed_filter[i];
    }
    gemm_cpu_noblas_par<float>(tmp, transformed_tile, A, 4, 2, 4); // 4*4*4*2 => M = 4, N=2, K=4 => 4*2
    gemm_cpu_noblas_par<float>(output, A_t, tmp, 2, 2, 4); // 2*4*4*2 => M = 2, N=2, K=4 => 2*2
    delete tmp;
}

float* winograd_cpu(float* input, float* filter, unsigned int w_input, unsigned int h_input, unsigned int w_filter, unsigned int h_filter) {
    float* transformed_filter = new float[4 * 4];
    float* transformed_tile = new float[4 * 4];

    float* G = new float[4 * 3] {
        1.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.0f, 0.0f, 1.0f
        };


    float* G_t = new float[3 * 4] {
        1.0f, 0.5f, 0.5f, 0.0f,
            0.0f, 0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.5f, 1.0f,
        };

    float* B = new float[4 * 4] {
        1.0f, 0.0f, -1.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 0.0f,
            0.0f, -1.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, -1.0f
        };

    float* B_t = new float[4 * 4] {
        1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, -1.0f, 1.0f,
            -1.0f, 1.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, -1.0f
        };

    float* A = new float[4 * 2] {
        1.0f, 0.0f,
            0.0f, -1.0f,
            0.0f, 0.0f,
            -1.0f, 0.0f
        };


    float* A_t = new float[2 * 4] {
        1.0f, 0.0f, 0.0f, -1.0f,
            0.0f, -1.0f, 0.0f, 0.0f
        };

    // Pre-computed G*g*G_t
    transform_filter(transformed_filter, filter, G, G_t);

    // Input tiles
    const unsigned int nb_tiles = w_input / 4 * h_input / 4;
    const unsigned int input_tile_size = 16;
    const unsigned int output_tile_size = 4;

    float* output = new float[output_tile_size * nb_tiles];

    for (int tile_idx = 0; tile_idx < nb_tiles; tile_idx++) {
        float* input_tile = input + tile_idx * input_tile_size;
        std::cout << "ROSDPFOSDPFDSPFOSDFPOSDFPSD" << std::endl;
        transform_input(transformed_tile, input_tile, B, B_t);

        // Output transformation + Hadamard product
        transform_output(output + tile_idx * output_tile_size, transformed_tile, transformed_filter, A, A_t);
    }
    return output;
}






