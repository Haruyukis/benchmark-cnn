#include "gtest/gtest.h"
#include "../src/winograd/winograd.cuh"

TEST(WinogradGPUTest, FetchInputTile) {
    float* input = new float[256];

    // Initialize the input with values from 0 to 63
    for (int i = 0; i < 256; i++) {
        input[i] = static_cast<float>(i);
    }

    // Print the input in 8x8 format
    for (int i = 0; i < 256; i++) {
        std::cout << input[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl; // Newline after every 8 elements
    }

    winograd_host(NULL, input, NULL, 16, 16, NULL, NULL);

    // Free allocated memory
    delete[] input;

    // Check if the result matches the expected output
    // EXPECT_TRUE(compare_arrays(C, expected, n)) << "Hadamard product failed!";
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
