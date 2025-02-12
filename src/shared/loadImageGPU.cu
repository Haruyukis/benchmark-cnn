// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "loadImageGPU.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void kernelLoadImageGPU(unsigned char* imgCharDevice, cuFloatComplex* imgFloatDevice, 
                                   int width, int height, int channels, int trueWidth, int trueHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ unsigned char sharedMem[];

    int paddedIndex = y * width + x;
    int trueIndex = y * trueWidth + x;
    
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int sharedOffset = threadId * channels;

    if (x < trueWidth && y < trueHeight) {
        for (int c = 0; c < channels; c++) {
            sharedMem[sharedOffset + c] = imgCharDevice[trueIndex * channels + c];
        }
    } else {
        for (int c = 0; c < channels; c++) {
            sharedMem[sharedOffset + c] = 0;  // Zero padding
        }
    }

    __syncthreads();
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            imgFloatDevice[c * width * height + paddedIndex] = 
                make_cuFloatComplex((float)sharedMem[sharedOffset + c], 0.0f);
        }
    }
}

__global__ void kernelLoadImageGPUf(unsigned char* imgCharDevice, float* imgFloatDevice, 
                                   int width, int height, int channels, int trueWidth, int trueHeight) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    extern __shared__ unsigned char sharedMem[];

    int paddedIndex = y * width + x;
    int trueIndex = y * trueWidth + x;
    
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int sharedOffset = threadId * channels;

    if (x < trueWidth && y < trueHeight) {
        for (int c = 0; c < channels; c++) {
            sharedMem[sharedOffset + c] = imgCharDevice[trueIndex * channels + c];
        }
    } else {
        for (int c = 0; c < channels; c++) {
            sharedMem[sharedOffset + c] = 0;  // Zero padding
        }
    }

    __syncthreads();
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            imgFloatDevice[c * width * height + paddedIndex] = (float) sharedMem[sharedOffset + c];
        }
    }
}

float* loadImageGPUf(const char* path, int* trueWidth, int* trueHeight, int* width, int* height, int* channels ){
    unsigned char* imgCharHost = stbi_load(path, trueWidth, trueHeight, channels,0);
    *width = 1<<(int) log2(*trueWidth-1)+1;
    *height = 1<<(int) log2(*trueHeight-1)+1;
    printf("Image chargée, width:%d, height:%d, channels:%d\n",*trueWidth, *trueHeight, *channels);

    // imgCharDevice
    unsigned char* imgCharDevice;
    cudaMalloc(&imgCharDevice, (*channels)*(*width)*(*height)*sizeof(unsigned char));
    cudaMemcpy(imgCharDevice, imgCharHost, (*channels)*(*width)*(*height)*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // imgFloatHost
    float* imgFloatHost = (float*) malloc((*channels)*(*width)*(*height)*sizeof(float));

    // imgFloatDevice
    float* imgFloatDevice;
    cudaMalloc(&imgFloatDevice, (*channels)*(*width)*(*height)*sizeof(float));

    int BLOCK_SIZE = 16;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((*width + blockSize.x - 1) / blockSize.x, (*height + blockSize.y - 1) / blockSize.y);

    int sharedMemorySize = BLOCK_SIZE * BLOCK_SIZE * *channels * sizeof(unsigned char);
    
    kernelLoadImageGPUf<<<gridSize, blockSize, sharedMemorySize>>>(imgCharDevice, imgFloatDevice, 
                                                                  *width, *height, *channels, 
                                                                  *trueWidth, *trueHeight);
    cudaDeviceSynchronize();
    printf("Image paddée, width:%d, height:%d, channels:%d\n",*width, *height, *channels);
    
    // Clean 
    stbi_image_free(imgCharHost);
    cudaFree(imgCharDevice);
    free(imgFloatHost);
    return imgFloatDevice;
}

cuFloatComplex* loadImageGPU(const char* path, int* trueWidth, int* trueHeight, int* width, int* height, int* channels ){
    unsigned char* imgCharHost = stbi_load(path, trueWidth, trueHeight, channels,0);
    *width = 1<<(int) log2(*trueWidth-1)+1;
    *height = 1<<(int) log2(*trueHeight-1)+1;
    printf("Image chargée, width:%d, height:%d, channels:%d\n",*trueWidth, *trueHeight, *channels);

    // imgCharDevice
    unsigned char* imgCharDevice;
    cudaMalloc(&imgCharDevice, (*channels)*(*width)*(*height)*sizeof(unsigned char));
    cudaMemcpy(imgCharDevice, imgCharHost, (*channels)*(*width)*(*height)*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // imgFloatHost
    cuFloatComplex* imgFloatHost = (cuFloatComplex*) malloc((*channels)*(*width)*(*height)*sizeof(cuFloatComplex));

    // imgFloatDevice
    cuFloatComplex* imgFloatDevice;
    cudaMalloc(&imgFloatDevice, (*channels)*(*width)*(*height)*sizeof(cuFloatComplex));

    int BLOCK_SIZE = 16;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((*width + blockSize.x - 1) / blockSize.x, (*height + blockSize.y - 1) / blockSize.y);

    int sharedMemorySize = BLOCK_SIZE * BLOCK_SIZE * *channels * sizeof(unsigned char);
    
    kernelLoadImageGPU<<<gridSize, blockSize, sharedMemorySize>>>(imgCharDevice, imgFloatDevice, 
                                                                  *width, *height, *channels, 
                                                                  *trueWidth, *trueHeight);
    cudaDeviceSynchronize();
    printf("Image paddée, width:%d, height:%d, channels:%d\n",*width, *height, *channels);
    
    // Clean 
    stbi_image_free(imgCharHost);
    cudaFree(imgCharDevice);
    free(imgFloatHost);
    return imgFloatDevice;
}

