#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <cuda_runtime.h>
#include "../shared/gemm.cuh"
#include "../shared/loadImageGPU.cu"
#include "../shared/storeImageGPU.cu"

using namespace std;

// Simple error-checking macro.
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << " at line "          \
                      << __LINE__ << ": " << cudaGetErrorString(err) << "\n";  \
            exit(err);                                                        \
        }                                                                     \
    }

//---------------------------------------------------------------------
// im2col_kernel: transforms the input image into a matrix where each 
// column is one 3x3 patch (across 3 channels) from the image.
// The input image is assumed to be stored in channels-first order
// (i.e., [channel][row][col]).
//---------------------------------------------------------------------
__global__ void im2col_kernel(float *image, float *cols,
                              int channels, int height, int width,
                              int kH, int kW, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPatches = outH * outW;
    if (idx < numPatches) {
        // Compute the top-left position of the patch.
        int w_out = idx % outW;
        int h_out = idx / outW;
        // For each channel and kernel element.
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < kH; i++) {
                for (int j = 0; j < kW; j++) {
                    int row = c * kH * kW + i * kW + j;
                    int in_row = h_out + i;
                    int in_col = w_out + j;
                    cols[row * numPatches + idx] =
                        image[c * height * width + in_row * width + in_col];
                }
            }
        }
    }
}

//---------------------------------------------------------------------
// gemm: A simple matrix multiplication kernel that computes:
//        C = A * B
// where A is (M x K), B is (K x N), and C is (M x N).
// In our convolution, A is the flattened kernel (size 1x27),
// B is the im2col matrix (27 x numPatches),
// and C is (1 x numPatches).
//---------------------------------------------------------------------
__global__ void gemm_debug(float *A, float *B, float *C,
                     int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

//---------------------------------------------------------------------
// conv_im2col_gemm: Host function to perform convolution using the
// im2col transformation and GEMM.
//  - d_image_in:  device pointer to input image (3, height, width)
//  - d_kernel:    device pointer to the kernel (flattened 27 floats)
//  - d_image_out: device pointer to output (size: (height-2) x (width-2))
//  - height, width: dimensions of the input image
//---------------------------------------------------------------------
void conv_im2col_gemm(float *d_image_in, float *d_kernel,
                      float *d_image_out, int height, int width)
{
    const int channels = 3;
    const int kH = 3, kW = 3;
    int outH = height - kH + 1;
    int outW = width - kW + 1;
    int numPatches = outH * outW;
    int kernelSize = channels * kH * kW; // 27

    // Allocate device memory for the im2col matrix.
    float *d_cols = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cols, kernelSize * numPatches * sizeof(float)));

    // Launch im2col kernel.
    int threadsPerBlock = 256;
    int numBlocks = (numPatches + threadsPerBlock - 1) / threadsPerBlock;
    im2col_kernel<<<numBlocks, threadsPerBlock>>>(d_image_in, d_cols,
                                                  channels, height, width,
                                                  kH, kW, outH, outW);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform GEMM: (1 x 27) * (27 x numPatches) = (1 x numPatches).
    // We use a 2D thread grid.
    int gemmBlockSize = 16;
    gemm(d_kernel, d_cols, d_image_out, kernelSize, 1, numPatches, gemmBlockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_cols));
}

//---------------------------------------------------------------------
// Main function:
//   - Loads "poupoupidou.jpg" using OpenCV.
//   - Converts the image to float and rearranges it from HWC (OpenCV)
//     to CHW order (required by the CUDA kernels).
//   - Sets up a 3x3x3 averaging kernel.
//   - Copies data to the GPU, runs the convolution, and copies the
//     result back.
//   - Saves the output as "output.jpg".
//---------------------------------------------------------------------
int main(int argc, char *argv[])
{
    // Load the image from file.
    if (argc != 2){
        fprintf(stderr, "Usage: %s <chemin_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* path = argv[1];
    int width, height, channels;
    float* d_image_in = loadImageGPUfGemm(path, &width, &height, &channels);

    std::cout << "Loaded image: " << path << " with size " 
              << width << "x" << height << std::endl;

    if (channels != 3) {
        std::cerr << "Error: Image must have 3 channels." << std::endl;
        return -1;
    }

    // Define kernel.
    const int kernelSize = 3 * 3 * 3; // 27 elements.
    float h_kernel[kernelSize];
    for (int i = 0; i < 3; i++) {
        h_kernel[i] = -1;
        h_kernel[i+9] = -1;
        h_kernel[i+18] = -1;
    }
    for (int i = 3; i < 6; i++) {
        h_kernel[i] = 0;
        h_kernel[i+9] = 0;
        h_kernel[i+18] = 0;
    }
    for (int i = 6; i < 9; i++) {
        h_kernel[i] = 1;
        h_kernel[i+9] = 1;
        h_kernel[i+18] = 1;
    }

    // Allocate device memory for the kernel and copy it.
    float *d_kernel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kernel, kernelSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernelSize * sizeof(float),
                            cudaMemcpyHostToDevice));

    // Determine the output dimensions.
    // With a 3x3 convolution (no padding, stride=1), the output is (height-2)x(width-2).
    int outH = height - 2;
    int outW = width - 2;
    size_t outputSize = outH * outW;

    // Allocate device memory for the output.
    float *d_image_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image_out, outputSize * sizeof(float)));

    // Run the convolution.
    conv_im2col_gemm(d_image_in, d_kernel, d_image_out, height, width);

    // // Copy the output back to host.
    // float *h_output = new float[outputSize];
    // CUDA_CHECK(cudaMemcpy(h_output, d_image_out, outputSize * sizeof(float),
    //                         cudaMemcpyDeviceToHost));

    // // Convert output to displayable format
    // float a = 1024; 
    // float b = -1024;
    // float pixel;
    // for (int i = 0; i < outW * outH; i++) {
    //     pixel = h_output[i];
    //     if (pixel < a) {
    //         a = pixel;
    //     } else if (pixel > b) {
    //         b = pixel;
    //     }
    // }

    // float div = b - a;
    // for (int i = 0; i < outW * outH; i++) {
    //     h_output[i] = 255 * (h_output[i] - a) / div;
    //     // printf("%f\n", h_output[i]);
    // }

    // CUDA_CHECK(cudaMemcpy(d_image_out, h_output, outputSize * sizeof(float),
    //                         cudaMemcpyHostToDevice));

    // Write the output image to file.
    const char* outFile = "../../out/output.jpg";
    storeImageGPUfGemm(d_image_out, outFile, outW, outH, 1);
    std::cout << "Output image saved as " << outFile << std::endl;

    // Free host and device memory.
    CUDA_CHECK(cudaFree(d_image_in));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_image_out));

    return 0;
}
