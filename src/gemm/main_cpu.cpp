#define STB_IMAGE_IMPLEMENTATION
#include <vector>
#include <iostream>
#include "../shared/loadImageCPU.cu"
#include "../shared/storeImageCPU.cu"

using namespace std;

void im2col_cpu(const std::vector<float>& image, std::vector<float>& cols,
                 int channels, int height, int width,
                 int kH, int kW, int outH, int outW) {
    int numPatches = outH * outW;
    cols.resize(channels * kH * kW * numPatches);

    for (int idx = 0; idx < numPatches; idx++) {
        int w_out = idx % outW;
        int h_out = idx / outW;
        
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

void gemm_cpu(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
               int M, int N, int K) {
    C.resize(M * N, 0.0f);
    
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void conv_im2col_gemm_cpu(const std::vector<float>& image, const std::vector<float>& kernel,
                           std::vector<float>& output, int height, int width) {
    const int channels = 3;
    const int kH = 3, kW = 3;
    int outH = height - kH + 1;
    int outW = width - kW + 1;
    int numPatches = outH * outW;
    int kernelSize = channels * kH * kW;

    std::vector<float> cols;
    im2col_cpu(image, cols, channels, height, width, kH, kW, outH, outW);
    
    gemm_cpu(kernel, cols, output, 1, numPatches, kernelSize);
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
    std::vector<float> h_image_in;
    int width, height, channels;
    loadImageCPU(path, h_image_in, &width, &height, &channels);

    std::cout << "Loaded image: " << path << " with size " 
              << width << "x" << height << std::endl;

    if (channels != 3) {
        std::cerr << "Error: Image must have 3 channels." << std::endl;
        return -1;
    }

    // Define kernel.
    const int kernelSize = 3 * 3 * 3; // 27 elements.
    std::vector<float> h_kernel(kernelSize, 0);
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

    // Determine the output dimensions.
    // With a 3x3 convolution (no padding, stride=1), the output is (height-2)x(width-2).
    int outH = height - 2;
    int outW = width - 2;
    size_t outputSize = outH * outW;
    std::vector<float> h_output;

    // Run the convolution.
    conv_im2col_gemm_cpu(h_image_in, h_kernel, h_output, height, width);

    // Write the output image to file.
    const char* outFile = "../../out/output.jpg";
    storeImageCPU(h_output, outFile, outW, outH, 1);
    std::cout << "Output image saved as " << outFile << std::endl;

    return 0;
}
