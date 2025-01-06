#include "gtest/gtest.h"
#include "../src/shared/hadamard.cuh"

bool compare_arrays(const double* A, const double* B, int n, double tolerance = 1e-5) {
    for (int i = 0; i < n; ++i) {
        if (abs(A[i] - B[i]) > tolerance) {
            return false;
        }
    }
    return true;
}


TEST(HadamardTest, VectorTest) {
    // Define two input vectors for the Hadamard product
    int n = 5;
    MatrixVo A = new double[n]; 
    MatrixVo B = new double[n]; 
    MatrixVo expected = new double[n];
    MatrixVo C = new double[n]; // Result matrix

    // Initialize Matrix A
    A[0] = 1.0f;
    A[1] = 2.0f;
    A[2] = 3.0f;
    A[3] = 4.0f;
    A[4] = 5.0f;

    // Initialize Matrix B
    B[0] = 5.0f;
    B[1] = 4.0f;
    B[2] = 3.0f;
    B[3] = 2.0f;
    B[4] = 1.0f;

    // Define expected output (element-wise multiplication)
    expected[0] = 5.0f;
    expected[1] = 8.0f;
    expected[2] = 9.0f;
    expected[3] = 8.0f;
    expected[4] = 5.0f;
    
    // Call the Hadamard function
    hadamard(C, A, B, n, 1);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(C, expected, n)) << "Hadamard product failed!";
}

TEST(HadamardTest, MatrixTest) {
    // Define two input vectors for the Hadamard product
    int n = 3;
    MatrixVo A = new double[9];  // Allocate a 3x3 matrix
    MatrixVo B = new double[9];  // Allocate a 3x3 matrix
    MatrixVo expected = new double[9];  // Allocate a 3x3 matrix
    MatrixVo C = new double[9];  // Result matrix
        
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
