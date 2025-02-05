#include "gtest/gtest.h"
#include "../src/winograd/winograd.hpp"
#include <iostream>

bool compare_arrays(const float* A, const float* B, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n * n; ++i) {
        if (abs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

TEST(WinogradTest, GG_TTest) {
    // Define G and G_transpose input vectors for filter transforme
    float* G = new float[4 * 3] {
        1.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.0f, 0.0f, 1.0f
        };

    float* G_transpose = new float[3 * 4] {
        1.0f, 0.5f, 0.5f, 0.0f,
            0.0f, 0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.5f, 1.0f,
        };

    int M = 4;
    int N = 4;
    int K = 3;
    float* C = new float[4 * 4];  // Result matrix
    float* expected = new float[4 * 4] {
        1.0f, 0.5f, 0.5f, 0.0f,
            0.5f, 0.75f, 0.25f, 0.5f,
            0.5f, 0.25f, 0.75f, 0.5f,
            0.0f, 0.5f, 0.5f, 1.0f
        };

    // Call the cpu gemm function
    gemm_cpu_noblas_par<float>(C, G, G_transpose, M, N, K);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(C, expected, 4)) << "Matrix Multiplication on CPU failed !";
}

TEST(WinogradTest, FilterTransformerTest) {
    // Define G and G_transpose input vectors for filter transforme
    float* G = new float[4 * 3] {
        1.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.0f, 0.0f, 1.0f
        };

    float* G_t = new float[3 * 4] {
        1.0f, 0.5f, 0.5f, 0.0f,
            0.0f, 0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.5f, 1.f,
        };

    float* filter = new float[3 * 3] {
        1.0f, 1.0f, 1.0f,
            -2.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f
        };

    float* expected = new float[4 * 4] {
        1.0f, 1.5f, 0.5f, 1.0f,
            0.0f, 1.5f, 0.0f, 1.5f,
            2.0f, 1.5f, 1.f, 0.5f,
            1.0f, 1.5f, 0.5f, 1.0f
        };

    float* transformed_filter = new float[4 * 4];

    // Call the cpu gemm function
    transform_filter(transformed_filter, filter, G, G_t);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(transformed_filter, expected, 4)) << "Matrix Multiplication on CPU failed !";
}

TEST(WinogradTest, InputTransformerTest) {
    // Define B and B_t input vectors for input transforme
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

    float* input = new float[4 * 4] {
        1.0f, 1.0f, 1.0f, 3.0f,
            -2.0f, 1.0f, 1.0f, -1.5f,
            1.0f, 1.0f, 1.0f, -2.5f,
            3.0f, 2.1f, 1.7f, -1.3f
        };

    float* expected = new float[4 * 4] {
        0.0f, 0.0f, 0.0f, -5.5f,
            -3.0f, 4.0f, 0.0f, 6.f,
            3.0f, 0.f, 0.f, 1.f,
            -4.3f, -1.8f, 0.4f, -0.9f
        };

    float* transformed_input = new float[4 * 4];

    // Call the cpu gemm function
    transform_input(transformed_input, input, B, B_t);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(transformed_input, expected, 4)) << "Matrix Multiplication on CPU failed !";
}

TEST(WinogradTest, InputTileSplit) {
    // Define 4 tiles
    const unsigned int nb_tiles = 4;
    const unsigned int tile_size = 16;

    float* input_tile1 = new float[4 * 4] {
        1.0f, 1.0f, 1.0f, 3.0f,
            -2.0f, 1.0f, 1.0f, -1.5f,
            1.0f, 1.0f, 1.0f, -2.5f,
            3.0f, 2.1f, 1.7f, -1.3f
        };

    float* input_tile2 = new float[4 * 4] {
        5.0f, 2.0f, 3.0f, 6.0f,
            -2.0f, 5.0f, 1.0f, -1.7f,
            1.0f, 1.0f, 1.0f, -2.6f,
            3.0f, 2.1f, 1.7f, -1.4f
        };

    float* input_tile3 = new float[4 * 4] {
        1.0f, 2.0f, 3.0f, 6.0f,
            -3.0f, 5.4f, 1.0f, -1.7f,
            1.0f, 6.5f, 2.1f, -2.6f,
            3.0f, 2.6f, 1.3f, -1.4f
        };

    float* input_tile4 = new float[4 * 4] {
        0.0f, 4.0f, 3.0f, 6.0f,
            -3.0f, 5.2f, 3.0f, -1.7f,
            1.0f, 2.5f, 2.7f, -2.6f,
            3.0f, 2.6f, 1.95f, -1.4f
        };

    float* tiles[nb_tiles] = { input_tile1, input_tile2, input_tile3, input_tile4 };

    float* input = new float[nb_tiles * tile_size];

    for (int tile_idx = 0; tile_idx < nb_tiles; tile_idx++) {
        for (int i = 0; i < tile_size; i++) {
            input[tile_idx * tile_size + i] = tiles[tile_idx][i];
        }
    }

    // Try retrieve tile per tile.
    float* retrieved_tile;
    for (int tile_idx = 0; tile_idx < nb_tiles; tile_idx++) {
        retrieved_tile = input + tile_idx * tile_size;
        EXPECT_TRUE(compare_arrays(retrieved_tile, tiles[tile_idx], 4)) << "Failed to retrieve the right tile !";
    }
}


TEST(WinogradTest, OutputTransformerTest) {
    const unsigned int nb_tiles = 4;
    const unsigned int tile_size = 16;

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

    float* G = new float[4 * 3] {
        1.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.0f, 0.0f, 1.0f
        };

    float* G_t = new float[3 * 4] {
        1.0f, 0.5f, 0.5f, 0.0f,
            0.0f, 0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.5f, 1.f,
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


    float* input = new float[4 * 4] {
        1.0f, 1.0f, 1.0f, 3.0f,
            -2.0f, 1.0f, 1.0f, -1.5f,
            1.0f, 1.0f, 1.0f, -2.5f,
            3.0f, 2.1f, 1.7f, -1.3f
        };

    float* filter = new float[3 * 3] {
        1.0f, 1.0f, 1.0f,
            -2.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f
        };

    float* output = new float[2 * 2];

    float* transformed_input_tile = new float[4 * 4];
    float* transformed_filter = new float[4 * 4];
    transform_input(transformed_input_tile, input, B, B_t);
    transform_filter(transformed_filter, filter, G, G_t);
    transform_output(output, transformed_input_tile, transformed_filter, A, A_t);
    float* expected = new float[2 * 2] {
        8.9f, -2.7f,
            9.0f, 6.0f
        };
    EXPECT_TRUE(compare_arrays(output, expected, 2)) << "Output for 1 tile failed !";

}

TEST(WinogradTest, WinogradMultipleTileTest) {
    const unsigned int nb_tiles = 4;
    const unsigned int input_tile_size = 16;
    const unsigned int output_tile_size = 4;

    float* input_tile1 = new float[4 * 4] {
        1.0f, 1.0f, 1.0f, 3.0f,
            -2.0f, 1.0f, 1.0f, -1.5f,
            1.0f, 1.0f, 1.0f, -2.5f,
            3.0f, 2.1f, 1.7f, -1.3f
        };

    float* input_tile2 = new float[4 * 4] {
        5.0f, 2.0f, 3.0f, 6.0f,
            -2.0f, 5.0f, 1.0f, -1.7f,
            1.0f, 1.0f, 1.0f, -2.6f,
            3.0f, 2.1f, 1.7f, -1.4f
        };

    float* input_tile3 = new float[4 * 4] {
        1.0f, 2.0f, 3.0f, 6.0f,
            -3.0f, 5.4f, 1.0f, -1.7f,
            1.0f, 6.5f, 2.1f, -2.6f,
            3.0f, 2.6f, 1.3f, -1.4f
        };

    float* input_tile4 = new float[4 * 4] {
        0.0f, 4.0f, 3.0f, 6.0f,
            -3.0f, 5.2f, 3.0f, -1.7f,
            1.0f, 2.5f, 2.7f, -2.6f,
            3.0f, 2.6f, 1.95f, -1.4f
        };

    float* tiles[nb_tiles] = { input_tile1, input_tile2, input_tile3, input_tile4 };

    float* input = new float[nb_tiles * input_tile_size];

    for (int tile_idx = 0; tile_idx < nb_tiles; tile_idx++) {
        for (int i = 0; i < input_tile_size; i++) {
            input[tile_idx * input_tile_size + i] = tiles[tile_idx][i];
        }
    }

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

    float* G = new float[4 * 3] {
        1.0f, 0.0f, 0.0f,
            0.5f, 0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            0.0f, 0.0f, 1.0f
        };

    float* G_t = new float[3 * 4] {
        1.0f, 0.5f, 0.5f, 0.0f,
            0.0f, 0.5f, -0.5f, 0.0f,
            0.0f, 0.5f, 0.5f, 1.f,
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

    float* filter = new float[3 * 3] {
        1.0f, 1.0f, 1.0f,
            -2.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f
        };
    float* expected = new float[4 * nb_tiles] {
        8.9f, -2.7f, 9.f, 6.f,
            17.1f, -1.2f, 15.45f, 12.f,
            21.f, 9.15f, 24.3f, 22.5f,
            15.75f, 2.775f, 18.f, 20.1f
        };


    float* output = winograd_cpu(input, filter, 8, 8, 3, 3);
    for (int i = 0; i < 4 * nb_tiles; i++) {
        std::cout << output[i] << " ";
        if ((i + 1) % 4 == 0) {
            std::cout << std::endl;
        }
    }
    EXPECT_TRUE(compare_arrays(output, expected, 4)) << "Output for 1 tile failed !";

}



int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
