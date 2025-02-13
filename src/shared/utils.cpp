#include "utils.hpp"

bool compare_arrays(const float* A, const float* B, int n, float tolerance) {
    for (int i = 0; i < n; ++i) {
        if (abs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

void printTile(float* input, int startRow, int startCol, int stride, int size, int width) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << input[(startRow + i) * width + (startCol + j)] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printMatrix(float* input, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << input[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}