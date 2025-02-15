#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

__global__ void kernelStoreImageGPU(cuFloatComplex* imgDevice, unsigned char* imgDeviceChar, int width, int height, int trueWidth, int trueHeight, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= trueWidth || y >= trueHeight) return;

    int pixelIdx = y * width + x; // with padding
    int truePixelIdx = y * trueWidth + x; // without padding

    if (channels == 1) {
        imgDeviceChar[pixelIdx] = (unsigned char) cuCabsf(imgDevice[pixelIdx]);
        
    } else if (channels == 3) {
        int rIdx = pixelIdx;
        int gIdx = pixelIdx + width * height;
        int bIdx = pixelIdx + 2 * width * height;
        int charIdx = (truePixelIdx) * channels;

        imgDeviceChar[charIdx]     = (unsigned char) cuCabsf(imgDevice[rIdx]);
        imgDeviceChar[charIdx + 1] = (unsigned char) cuCabsf(imgDevice[gIdx]);
        imgDeviceChar[charIdx + 2] = (unsigned char) cuCabsf(imgDevice[bIdx]);
    }
}

__global__ void kernelStoreImageGPUf(float* imgDevice, unsigned char* imgDeviceChar, int width, int height, int trueWidth, int trueHeight, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= trueWidth || y >= trueHeight) return;

    int pixelIdx = y * width + x; // with padding
    int truePixelIdx = y * trueWidth + x; // without padding

    if (channels == 1) {
        imgDeviceChar[pixelIdx] = (unsigned char) imgDevice[pixelIdx];
        
    } else if (channels == 3) {
        int rIdx = pixelIdx;
        int gIdx = pixelIdx + width * height;
        int bIdx = pixelIdx + 2 * width * height;
        int charIdx = (truePixelIdx) * channels;

        imgDeviceChar[charIdx]     = (unsigned char) imgDevice[rIdx];
        imgDeviceChar[charIdx + 1] = (unsigned char) imgDevice[gIdx];
        imgDeviceChar[charIdx + 2] = (unsigned char) imgDevice[bIdx];
    }
}

__global__ void kernelStoreImageGPUfGemm(float* imgDevice, unsigned char* imgDeviceChar, int trueWidth, int trueHeight, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= trueWidth || y >= trueHeight) return;

    int pixelIdx = y * trueWidth + x;

    if (channels == 1) {
        imgDeviceChar[pixelIdx] = (unsigned char) imgDevice[pixelIdx];
    } else if (channels == 3) {
        int rIdx = pixelIdx;
        int gIdx = pixelIdx + trueWidth * trueHeight;
        int bIdx = pixelIdx + 2 * trueWidth * trueHeight;
        int charIdx = pixelIdx * channels;

        imgDeviceChar[charIdx]     = (unsigned char) imgDevice[rIdx];
        imgDeviceChar[charIdx + 1] = (unsigned char) imgDevice[gIdx];
        imgDeviceChar[charIdx + 2] = (unsigned char) imgDevice[bIdx];
    }
}

void storeImageGPUfGemm(float* imgDevice, const char* path, int trueWidth, int trueHeight, int channels) {
    unsigned char* imgDeviceChar;
    cudaMalloc(&imgDeviceChar, trueWidth * trueHeight * channels * sizeof(unsigned char));

    dim3 blockSize(16, 16);
    dim3 gridSize((trueWidth + blockSize.x - 1) / blockSize.x, 
                  (trueHeight + blockSize.y - 1) / blockSize.y);

    kernelStoreImageGPUfGemm<<<gridSize, blockSize>>>(imgDevice, imgDeviceChar, trueWidth, trueHeight, channels);
    cudaDeviceSynchronize();

    unsigned char* imgHostChar = (unsigned char*) malloc(trueWidth * trueHeight * channels * sizeof(unsigned char));
    cudaMemcpy(imgHostChar, imgDeviceChar, trueWidth * trueHeight * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_jpg(path, trueWidth, trueHeight, channels, imgHostChar, 90);
    
    cudaFree(imgDeviceChar);
    free(imgHostChar);
}



void storeImageGPU(cuFloatComplex* imgDevice, const char* path, int trueWidth, int trueHeight, int width, int height, int channels ){
    unsigned char* imgDeviceChar;
    cudaMalloc(&imgDeviceChar, trueWidth*trueHeight*channels*sizeof(unsigned char));
    // appel kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                (height + blockSize.y - 1) / blockSize.y);

    kernelStoreImageGPU<<<gridSize, blockSize>>>(imgDevice, imgDeviceChar, width, height, trueWidth, trueHeight, channels);
    cudaDeviceSynchronize();

    unsigned char* imgHostChar = (unsigned char*) malloc(trueWidth*trueHeight*channels*sizeof(unsigned char));
    cudaMemcpy(imgHostChar, imgDeviceChar, trueWidth*trueHeight*channels*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_jpg(path, trueWidth, trueHeight, channels, imgHostChar, 90);

    // Clean :
    cudaFree(imgDeviceChar);
    free(imgHostChar);
}

void storeImageGPUf(float* imgDevice, const char* path, int trueWidth, int trueHeight, int width, int height, int channels ){
    unsigned char* imgDeviceChar;
    cudaMalloc(&imgDeviceChar, trueWidth*trueHeight*channels*sizeof(unsigned char));
    // appel kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                (height + blockSize.y - 1) / blockSize.y);

    kernelStoreImageGPUf<<<gridSize, blockSize>>>(imgDevice, imgDeviceChar, width, height, trueWidth, trueHeight, channels);
    cudaDeviceSynchronize();

    unsigned char* imgHostChar = (unsigned char*) malloc(trueWidth*trueHeight*channels*sizeof(unsigned char));
    cudaMemcpy(imgHostChar, imgDeviceChar, trueWidth*trueHeight*channels*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_jpg(path, trueWidth, trueHeight, channels, imgHostChar, 90);
    
    // Clean :
    cudaFree(imgDeviceChar);
    free(imgHostChar);
}

#include <iostream>
#include <vector>
#include "stb_image_write.h"