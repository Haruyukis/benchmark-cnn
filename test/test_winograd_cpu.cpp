#include "gtest/gtest.h"
#include "../src/winograd/winograd.hpp"
#include <iostream>
#include "../src/shared/utils.hpp"

TEST(WinogradTest, FilterTransformerTest)
{
    // Define G and G_transpose input vectors for filter transforme
    float *G = new float[4 * 3]{
        1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.5f,
        0.5f, -0.5f, 0.5f,
        0.0f, 0.0f, 1.0f};

    float *G_t = new float[3 * 4]{
        1.0f,
        0.5f,
        0.5f,
        0.0f,
        0.0f,
        0.5f,
        -0.5f,
        0.0f,
        0.0f,
        0.5f,
        0.5f,
        1.f,
    };

    float *filter = new float[3 * 3]{
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f,
        1.0f, 0.0f, -1.0f};

    float *expected = new float[16]{1.0f, 0.f, 0.f, -1.0f,
                                    1.5f, 0.f, 0.f, -1.5f,
                                    0.5f, 0.f, 0.f, -0.5f,
                                    1.0f, 0.f, 0.f, -1.0f};

    float *transformed_filter = new float[4 * 4];

    // Call the cpu gemm function
    transform_filter(transformed_filter, filter, G, G_t);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(transformed_filter, expected, 4, 1e-5)) << "Matrix Multiplication on CPU failed !";

    delete[] expected, filter, G, G_t, transformed_filter;
}

TEST(WinogradTest, InputTransformerTest)
{
    // Define B and B_t input vectors for input transforme
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

    float *input_tile = new float[4 * 4]{
        0.0f, 1.0f, 2.0f, 3.0f,
        10.0f, 11.0f, 12.0f, 13.0f,
        20.0f, 21.0f, 22.0f, 23.f,
        30.0f, 31.f, 32.f, 33.f};

    float *expected = new float[4 * 4]{
        0.f, -40.f, 0.f, 0.f,
        -4.f, 66.f, 2.f, -4.f,
        0.f, 20.f, 0.f, 0.f,
        0.f, -40.f, 0.f, 0.f};

    float *transformed_input = new float[4 * 4];

    // Call the cpu gemm function
    transform_input(transformed_input, input_tile, B, B_t);

    // Check if the result matches the expected output
    EXPECT_TRUE(compare_arrays(transformed_input, expected, 4, 1e-5))
        << "Input Transformation on CPU failed !";

    delete[] B, B_t, input_tile, expected, transformed_input;
}

TEST(WinogradCPUTest, Winograd10x10Test)
{
    int width = 10;
    int height = 10;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *input = new float[i_size];
    float *output = new float[o_size];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++)
    {
        input[i] = static_cast<float>(i);
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "10x10 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] filter;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, Winograd16x16Test)
{
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *input = new float[i_size];
    float *output = new float[o_size];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++)
    {
        input[i] = static_cast<float>(i);
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "16x16 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] filter;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, Winograd32x32Test)
{
    int width = 32;
    int height = 32;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *input = new float[i_size];
    float *output = new float[o_size];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++)
    {
        input[i] = static_cast<float>(i);
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "32x32 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] filter;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, Winograd64x64Test)
{
    int width = 64;
    int height = 64;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *input = new float[i_size];
    float *output = new float[o_size];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++)
    {
        input[i] = static_cast<float>(i);
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "64x64 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] filter;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, Winograd128x128Test)
{
    int width = 128;
    int height = 128;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *input = new float[i_size];
    float *output = new float[o_size];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int i = 0; i < i_size; i++)
    {
        input[i] = static_cast<float>(i);
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, 1);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    EXPECT_TRUE(compare_arrays(output, expected, o_width, 1e-5)) << "128x128 tiles input convolution failed!";
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] filter;
    delete[] expected;

    // Check if the result matches the expected output
}

// RGB Channel same kernel
TEST(WinogradCPUTest, RGBWinograd10x10Test)
{
    int width = 10;
    int height = 10;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float *input = new float[i_size * nb_channel];
    float *output = new float[o_size * nb_channel];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++)
    {
        for (int i = 0; i < i_size; i++)
        {
            input[i + c * i_size] = static_cast<float>(i);
        }
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    for (int c = 0; c < nb_channel; c++)
    {
        EXPECT_TRUE(compare_arrays(output + c * o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, RGBWinograd16x16Test)
{
    int width = 16;
    int height = 16;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float *input = new float[i_size * nb_channel];
    float *output = new float[o_size * nb_channel];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++)
    {
        for (int i = 0; i < i_size; i++)
        {
            input[i + c * i_size] = static_cast<float>(i);
        }
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    for (int c = 0; c < nb_channel; c++)
    {
        EXPECT_TRUE(compare_arrays(output + c * o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, RGBWinograd32x32Test)
{
    int width = 32;
    int height = 32;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float *input = new float[i_size * nb_channel];
    float *output = new float[o_size * nb_channel];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++)
    {
        for (int i = 0; i < i_size; i++)
        {
            input[i + c * i_size] = static_cast<float>(i);
        }
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    for (int c = 0; c < nb_channel; c++)
    {
        EXPECT_TRUE(compare_arrays(output + c * o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, RGBWinograd64x64Test)
{
    int width = 64;
    int height = 64;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float *input = new float[i_size * nb_channel];
    float *output = new float[o_size * nb_channel];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++)
    {
        for (int i = 0; i < i_size; i++)
        {
            input[i + c * i_size] = static_cast<float>(i);
        }
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    for (int c = 0; c < nb_channel; c++)
    {
        EXPECT_TRUE(compare_arrays(output + c * o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

TEST(WinogradCPUTest, RGBWinograd128x128Test)
{
    int width = 128;
    int height = 128;
    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    int nb_channel = 3;

    float *input = new float[i_size * nb_channel];
    float *output = new float[o_size * nb_channel];
    float *filter = new float[9]{
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f,
        1.0f, 0.f, -1.f};

    // Initialize the input with values from 0 to i_size
    for (int c = 0; c < nb_channel; c++)
    {
        for (int i = 0; i < i_size; i++)
        {
            input[i + c * i_size] = static_cast<float>(i);
        }
    }

    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channel);

    float *expected = new float[o_size];
    for (int i = 0; i < o_size; i++)
    {
        expected[i] = -6.f;
    }

    for (int c = 0; c < nb_channel; c++)
    {
        EXPECT_TRUE(compare_arrays(output + c * o_size, expected, o_width, 1e-5)) << width << "x" << height << "tiles input convolution failed!";
    }
    // Free allocated memory
    delete[] input;
    delete[] output;
    delete[] expected;

    // Check if the result matches the expected output
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
