#include <cuComplex.h>

#ifndef LOAD_IMAGE_GPU_CUH
#define LOAD_IMAGE_GPU_CUH

cuFloatComplex* loadImageGPU(const char* path, int* trueWidth, int* trueHeight, int* width, int* height, int* channels);

float* loadImageGPUf(const char* path, int* width, int* height, int* channels);

__global__ void kernelLoadImageGPUf(unsigned char* imgCharDevice, float* imgFloatDevice, int width, int height, int nb_channels);

#endif