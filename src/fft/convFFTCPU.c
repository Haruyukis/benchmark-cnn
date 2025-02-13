#define STB_IMAGE_IMPLEMENTATION
#include "../shared/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../shared/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>

// Function to save an image from FFT data (real part only)
void save_image_as_jpeg(const char *filename, fftw_complex *data_r, fftw_complex *data_g, fftw_complex *data_b, int width, int height) {
    unsigned char *image = (unsigned char*)malloc(3 * width * height);
    
    // Find min/max for normalization
    // double min_val = 1e9, max_val = -1e9;
    // for (int i = 0; i < width * height; i++) {
    //     double real_value = data_r[i][0];  // Use only real part of Red channel
    //     if (real_value > max_val) max_val = real_value;
    //     if (real_value < min_val) min_val = real_value;
    // }

    // Normalize to 0-255
    for (int i = 0; i < width * height; i++) {
        image[3 * i] = (unsigned char)(data_r[i][0]/(width*height));
        image[3 * i + 1] = (unsigned char) (data_g[i][0]/(width*height) );
        image[3 * i + 2] = (unsigned char) (data_b[i][0]/(width*height) );
    }

    // Save as JPEG
    stbi_write_jpg(filename, width, height, 3, image, 90);
    free(image);
    printf("Saved image as JPEG: %s\n", filename);
}

// Function to compute FFT
void compute_fft(fftw_complex *input, fftw_complex *output, int width, int height) {
    fftw_plan plan = fftw_plan_dft_2d(height, width, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

// Function to compute Inverse FFT (IFFT)
void compute_ifft(fftw_complex *input, fftw_complex *output, int width, int height) {
    fftw_plan plan = fftw_plan_dft_2d(height, width, input, output, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

// Function to perform element-wise multiplication of two FFT outputs
void multiply_fft(fftw_complex *fft1, fftw_complex *fft2, fftw_complex *result, int size) {
    for (int i = 0; i < size; i++) {
        double real1 = fft1[i][0];
        double imag1 = fft1[i][1];
        double real2 = fft2[i][0];
        double imag2 = fft2[i][1];

        // Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
        result[i][0] = real1 * real2 - imag1 * imag2;
        result[i][1] = real1 * imag2 + imag1 * real2;
    }
}

int main() {
    int width, height, channels;

    // Load image as RGB
    unsigned char *image = stbi_load("Bayeux-tapisserie-guillaume-le-conquérant.jpg", &width, &height, &channels, 3);
    if (!image) {
        printf("Error loading image\n");
        return 1;
    }

    printf("Loaded image: %dx%d\n", width, height);

    int size = width * height;

    // Allocate FFTW memory for each RGB channel
    fftw_complex *input_image_r, *output_image_r;
    fftw_complex *input_image_g, *output_image_g;
    fftw_complex *input_image_b, *output_image_b;

    fftw_complex *multiplied_fft_r, *ifft_output_r;
    fftw_complex *multiplied_fft_g, *ifft_output_g;
    fftw_complex *multiplied_fft_b, *ifft_output_b;

    fftw_complex *input_kernel_r, *output_kernel_r;

    input_image_r = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    output_image_r = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    input_image_g = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    output_image_g = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    input_image_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    output_image_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    multiplied_fft_r = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    ifft_output_r = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    multiplied_fft_g = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    ifft_output_g = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    multiplied_fft_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    ifft_output_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    input_kernel_r = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    output_kernel_r = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * size);
    for (int i = 0; i < size; i++) {
        input_image_r[i][0] = (double)image[3 * i];
        input_image_r[i][1] = 0.0;
        input_image_g[i][0] = (double)image[3 * i + 1];
        input_image_g[i][1] = 0.0;
        input_image_b[i][0] = (double)image[3 * i + 2];
        input_image_b[i][1] = 0.0;
    }
    input_kernel_r[0 + 0*width][0] = (double) -1;
    input_kernel_r[1 + 0*width][0] = (double) -1;
    input_kernel_r[2 + 0*width][0] = (double) -1;

    input_kernel_r[0 + 1*width][0] = (double) 0;
    input_kernel_r[1 + 1*width][0] = (double) 0;
    input_kernel_r[2 + 1*width][0] = (double) 0;

    input_kernel_r[0 + 2*width][0] = (double) 1;
    input_kernel_r[1 + 2*width][0] = (double) 1;
    input_kernel_r[2 + 2*width][0] = (double) 1;

    // Compute FFT, Multiply, Compute IFFT for each channel
    compute_fft(input_image_r, output_image_r, width, height);
    compute_fft(input_image_g, output_image_g, width, height);
    compute_fft(input_image_b, output_image_b, width, height);

    compute_fft(input_kernel_r, output_kernel_r, width, height);
    compute_fft(input_kernel_r, output_kernel_r, width, height);
    compute_fft(input_kernel_r, output_kernel_r, width, height);

    multiply_fft(output_image_r, output_kernel_r, multiplied_fft_r, size);
    multiply_fft(output_image_g, output_kernel_r, multiplied_fft_g, size);
    multiply_fft(output_image_b, output_kernel_r, multiplied_fft_b, size);

    compute_ifft(multiplied_fft_r, ifft_output_r, width, height);
    compute_ifft(multiplied_fft_g, ifft_output_g, width, height);
    compute_ifft(multiplied_fft_b, ifft_output_b, width, height);

    save_image_as_jpeg("ifft_output.jpg", ifft_output_r, ifft_output_g, ifft_output_b, width, height);

    stbi_image_free(image);
    return 0;
}

// gcc src/fft/convFFTCPU.c -lfftw3 -lm -o build/convFFTCPU

