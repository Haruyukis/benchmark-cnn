#include "gtest/gtest.h"
#include "src/shared/hadamard.cu"

TEST(HadamardTest, BasicTest) {
    // Define two input vectors for the Hadamard product
    int n = 5;
    float A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float B[] = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    
    // Define expected output (element-wise multiplication)
    float expected[] = {5.0f, 8.0f, 9.0f, 8.0f, 5.0f};
    
    // Allocate space for the result on the host
    float C[n];
    
    // Call the Hadamard function
    hadamard(A, B, C, n);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(C, expected, n)) << "Hadamard product failed!";
}