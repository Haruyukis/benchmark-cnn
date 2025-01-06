#include "gtest/gtest.h"
#include "../src/shared/hadamard.cuh"

bool compare_arrays(const float* A, const float* B, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n; ++i) {
        if (abs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

TEST(HadamardTest, MatrixTest) {
    // Define two input vectors for the Hadamard product
    int n = 3;
    float* A = new float[9];  // Allocate a 3x3 matrix
    float* B = new float[9];  // Allocate a 3x3 matrix
    float* expected = new float[9];  // Allocate a 3x3 matrix
    float* C = new float[9];  // Result matrix
        
    // Initialize Matrix A
    A[0] = 1.0f; A[1] = 2.0f; A[2] = 3.0f;
    A[3] = 4.0f; A[4] = 5.0f; A[5] = 6.0f;
    A[6] = 7.0f; A[7] = 8.0f; A[8] = 9.0f;

    // Initialize Matrix B
    B[0] = 9.0f; B[1] = 8.0f; B[2] = 7.0f;
    B[3] = 6.0f; B[4] = 5.0f; B[5] = 4.0f;
    B[6] = 3.0f; B[7] = 2.0f; B[8] = 1.0f;

    // Expected result (element-wise multiplication)
    expected[0] = 9.0f; expected[1] = 16.0f; expected[2] = 21.0f;
    expected[3] = 24.0f; expected[4] = 25.0f; expected[5] = 24.0f;
    expected[6] = 21.0f; expected[7] = 16.0f; expected[8] = 9.0f;

    // Call the Hadamard function
    hadamard(C, A, B, n, n);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(C, expected, n)) << "Hadamard product failed!";
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
