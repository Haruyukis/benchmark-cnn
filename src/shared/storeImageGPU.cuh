#include <cuda_runtime.h>
#include <cuComplex.h>




void storeImageGPU(cuFloatComplex* imgDevice, const char* path, int trueWidth, int trueHeight, int width, int height, int channels );

void storeImageGPUf(float* imgDevice, const char* path, int trueWidth, int trueHeight, int width, int height, int channels);