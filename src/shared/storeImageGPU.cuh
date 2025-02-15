#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef STORE_IMAGE_GPU_CUH
#define STORE_IMAGE_GPU_CUH


void storeImageGPU(cuFloatComplex* imgDevice, const char* path, int trueWidth, int trueHeight, int width, int height, int channels );

void storeImageGPUf(float* imgDevice, const char* path, int w_output, int h_output, int nb_channels);

__global__ void kernelStoreImageGPUf(float* imgDevice, unsigned char* imgDeviceChar, int w_output, int h_output, int nb_channels);

#endif