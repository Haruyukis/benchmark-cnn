#include "winograd.hpp"
#include <iostream>
#include <utils.cpp>

/**
 * Transforming the input_tile into the transformed one (4x4 Matrix).
 * transformed_tile: contains the operation B*input_tile*B_t.
 * input_tile: the tile that is transformed (4x4)
 */
void transform_input(float *transformed_tile, float *input_tile, float *B, float *B_t)
{
    float *tmp = new float[4 * 4];
    gemm_cpu_noblas_par<float>(tmp, input_tile, B, 4, 4, 4);
    gemm_cpu_noblas_par<float>(transformed_tile, B_t, tmp, 4, 4, 4);
    delete tmp;
};

/**
 * Transform the filter for Winograd (4x4 Matrix).
 * transformed_filter: contains the operation G*filter*G_t.
 * filter: filter that is transformed (3x3 Filter)
 */

// TODO Give tmp as a parameter to avoid copy ?
void transform_filter(float *transformed_filter, float *filter, float *G, float *G_t)
{
    float *tmp = new float[3 * 4];
    gemm_cpu_noblas_par<float>(tmp, filter, G_t, 3, 4, 3);
    gemm_cpu_noblas_par<float>(transformed_filter, G, tmp, 4, 4, 3);
    delete tmp;
}

void transform_output(float *output, float *transformed_tile, float *transformed_filter, float *A, float *A_t)
{
    float *tmp = new float[4 * 4];
    // Hadamard Product
    for (int i = 0; i < 16; i++)
    {
        transformed_tile[i] = transformed_tile[i] * transformed_filter[i];
    }
    gemm_cpu_noblas_par<float>(tmp, transformed_tile, A, 4, 2, 4); // 4*4*4*2 => M = 4, N=2, K=4 => 4*2
    gemm_cpu_noblas_par<float>(output, A_t, tmp, 2, 2, 4);         // 2*4*4*2 => M = 2, N=2, K=4 => 2*2
    delete tmp;
}

void extract_tile(float *input, float *input_tile, int startRow, int startCol, int w_input, int h_input, int w_tile, int h_tile)
{
    for (int i = 0; i < h_tile; i++)
    {
        for (int j = 0; j < w_tile; j++)
        {
            input_tile[i * 4 + j] = input[(startRow + i) * w_input + (startCol + j)];
        }
    }
}

void winograd_cpu(float *output, float *input, float *filter, int w_input, int h_input, int w_filter, int h_filter, int nb_channel)
{
    float *transformed_filter = new float[16];
    float *transformed_tile = new float[16];

    float *input_tile = new float[16];
    float *output_tile = new float[4];

    int w_output = w_input - w_filter + 1;
    int h_output = h_input - h_filter + 1;

    float *G = new float[4 * 3]{
        1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f,
        0.5f, -0.5f, 0.5f,
        0.0f, 0.0f, 1.0f};

    float *G_t = new float[3 * 4]{
        1.0f, 0.5f, 0.5f, 0.0f,
        0.0f, 0.5f, -0.5f, 0.0f,
        0.0f, 0.5f, 0.5f, 1.0f};

    float *B = new float[4 * 4]{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, -1.0f, 1.0f,
        -1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, -1.0f};

    float *B_t = new float[4 * 4]{
        1.0f, 0.0f, -1.0f, 0.0f,
        0.0f, 1.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, -1.0f};

    float *A = new float[4 * 2]{
        1.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, -1.0f,
        0.0f, -1.0f};

    float *A_t = new float[2 * 4]{
        1.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 1.0f, -1.0f, -1.0f};

    //     // Pre-computed G*g*G_t
    transform_filter(transformed_filter, filter, G, G_t);

    int o_offset = w_output * h_output;
    int i_offset = w_input * h_input;

    for (int c = 0; c < nb_channel; c++)
    {
        float *input_c = input + c * i_offset;
        float *output_c = output + c * o_offset;
        for (int startRow = 0; startRow <= h_input - 4; startRow += 2)
        {
            for (int startCol = 0; startCol <= w_input - 4; startCol += 2)
            {
                extract_tile(input_c, input_tile, startRow, startCol, w_input, h_input, 4, 4);
                transform_input(transformed_tile, input_tile, B, B_t);

                transform_output(output_tile, transformed_tile, transformed_filter, A, A_t);

                // Write the 2x2 output_tile into the output at the right position
                int tile_idx = startRow * w_output + startCol;
                output_c[tile_idx] = output_tile[0];
                output_c[tile_idx + 1] = output_tile[1];
                output_c[tile_idx + w_output] = output_tile[2];
                output_c[tile_idx + w_output + 1] = output_tile[3];
            }
        }
    }

    delete[] G, G_t, A, A_t, B, B_t, transformed_filter, transformed_tile;
}

// float *winograd_cpu(float *output, float *input, float *filter, unsigned int w_input, unsigned int h_input, unsigned int w_filter, unsigned int h_filter)
// {
//     float *transformed_filter = new float[4 * 4];
//     float *transformed_tile = new float[4 * 4];

//     float *G = new float[4 * 3]{
//         1.0f, 0.0f, 0.0f,
//         0.5f, 0.5f, 0.5f,
//         0.5f, -0.5f, 0.5f,
//         0.0f, 0.0f, 1.0f};

//     float *G_t = new float[3 * 4]{
//         1.0f, 0.5f, 0.5f, 0.0f,
//         0.0f, 0.5f, -0.5f, 0.0f,
//         0.0f, 0.5f, 0.5f, 1.0f};

//     float *B = new float[4 * 4]{
//         1.0f, 0.0f, -1.0f, 0.0f,
//         0.0f, 1.0f, 1.0f, 0.0f,
//         0.0f, -1.0f, 1.0f, 0.0f,
//         0.0f, 1.0f, 0.0f, -1.0f};

//     float *B_t = new float[4 * 4]{
//         1.0f, 0.0f, 0.0f, 0.0f,
//         0.0f, 1.0f, -1.0f, 1.0f,
//         -1.0f, 1.0f, 1.0f, 0.0f,
//         0.0f, 0.0f, 0.0f, -1.0f};

//     float *A = new float[4 * 2]{
//         1.0f, 0.0f,
//         0.0f, -1.0f,
//         0.0f, 0.0f,
//         -1.0f, 0.0f};

//     float *A_t = new float[2 * 4]{
//         1.0f, 0.0f, 0.0f, -1.0f,
//         0.0f, -1.0f, 0.0f, 0.0f};

//     // Pre-computed G*g*G_t
//     transform_filter(transformed_filter, filter, G, G_t);

//     // Input tiles
//     const unsigned int nb_tiles = w_input / 4 * h_input / 4;
//     const unsigned int input_tile_size = 16;
//     const unsigned int output_tile_size = 4;

//     float output[4];

//     float input_tile[16];

//     for (int startRow = 0; startRow <= h_input - 4; startRow += 2)
//     {
//         for (int startCol = 0; startCol <= w_input - 4; startCol += 2)
//         {
//             // Directly reference the tile in input
//             for (int i = 0; i < 4; i++)
//             {
//                 for (int j = 0; j < 4; j++)
//                 {
//                     input_tile[i * 4 + j] = input[(startRow + i) * w_input + (startCol + j)];
//                 }
//                 std::cout << std::endl;
//             }

//             std::cout << "Tile n° (" << startRow << "," << startCol << ")" << std::endl;
//             for (int i = 0; i < 4; i++)
//             {
//                 for (int j = 0; j < 4; j++)
//                 {
//                     std::cout << input_tile[i * 4 + j] << " ";
//                 }
//                 std::cout << std::endl;
//             }

//             transform_input(transformed_tile, input_tile, B, B_t);

//             // // Output transformation + Hadamard product
//             transform_output(output, transformed_tile, transformed_filter, A, A_t);
//             for (int i = 0; i < 2; i++)
//             {
//                 for (int j = 0; j < 2; j++)
//                 {
//                 }
//             }
//         }
//     }

//     delete[] transformed_filter;
//     delete[] transformed_tile;

//     return output;
// }