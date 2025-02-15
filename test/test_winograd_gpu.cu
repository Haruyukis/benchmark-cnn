#include "gtest/gtest.h"
#include "../src/winograd/winograd.cuh"
#include "../src/shared/utils.hpp"


TEST(WinogradGPUTest, Winograd10x10Test) {
    int width = 10;
    int height = 10;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };
    
    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 8, 1e-5)) << "10x10 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd16x16Test) {
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 14, 1e-5)) << "16x16 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd32x32Test) {
    int width = 32;
    int height = 32;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 30, 1e-5)) << "32x32 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd64x64Test) {
    int width = 64;
    int height = 64;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 62, 1e-5)) << "64x64 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, Winograd128x128Test) {
    int width = 128;
    int height = 128;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float* input = new float[i_size];
    float* output = new float[o_size];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++) {
        input[i] = static_cast<float>(i);
    }

    winograd_host(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, 126, 1e-5)) << "128x128 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

// RGB Channel same kernel
TEST(WinogradGPUTest, RGBWinograd10x10Test) {
    int width = 10;
    int height = 10;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;


    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };
    
    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd16x16Test) {
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd32x32Test) {
    int width = 32;
    int height = 32;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd64x64Test) {
    int width = 64;
    int height = 64;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradGPUTest, RGBWinograd128x128Test) {
    int width = 128;
    int height = 128;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float* input = new float[i_size * nb_channel];
    float* output = new float[o_size * nb_channel];
    float* filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f
    };

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++){
        for (int i = 0; i < i_size; i++) {
            input[i + c*i_size] = static_cast<float>(i);
        }
    }

    winograd_host(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i=0; i<o_size; i++){
        expected[i] = -6.f;
    }

    for (int c=0; c<nb_channel; c++){
        EXPECT_TRUE(compare_arrays(output + c*o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
