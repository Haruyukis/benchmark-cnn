#include <cuComplex.h>

cuFloatComplex* loadImageGPU(const char* path, int* trueWidth, int* trueHeight, int* width, int* height, int* channels);

float* loadImageGPUf(const char* path, int* trueWidth, int* trueHeight, int* width, int* height, int* channels);