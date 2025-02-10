#include "gtest/gtest.h"
#include "../src/winograd/winograd.cuh"


void printTile(float* input, int startRow, int startCol, int stride, int size, int width) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << input[(startRow + i) * width + (startCol + j)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


TEST(WinogradGPUTest, FetchInputTile) {
    int width = 16; // Adjust width for 256 elements
    int height = 16; // Assuming a square 16x16 input
    float* input = new float[256];
    float* output = new float[256];
    
    // Initialize the input with values from 0 to 255
    for (int i = 0; i < 256; i++) {
        input[i] = static_cast<float>(i);
    }

    // Print the 4x4 tiles with stride 2
    for (int row = 0; row <= height - 4; row += 2) {
        for (int col = 0; col <= width - 4; col += 2) {
            std::cout << "Tile at (" << row << ", " << col << "):" << std::endl;
            printTile(input, row, col, 2, 4, width);
        }
    }

    std::cout << "--------------------------- GPU ----------------" << std::endl;

    winograd_host(output, input, NULL, width, height, 3, 3);
    for (int i=0; i < 16; i++){
        for (int j = 0; j < 16; j++){
            std::cout << static_cast<int>(output[i * 16 + j]) << " ";
        }
        std::cout << std::endl;
    }

    // Free allocated memory
    delete[] input;

    // Check if the result matches the expected output
    // EXPECT_TRUE(compare_arrays(C, expected, n)) << "Hadamard product failed!";
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
