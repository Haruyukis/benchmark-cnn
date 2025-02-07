#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef FFTSHAREDROW
#define FFTSHAREDROW

__global__ void fft_DIF_on_rows(cuFloatComplex* image_float, int width, int height, int log2width);

__global__ void ifft_DIT_on_rows(cuFloatComplex* image_float, int width, int height, int log2width);

#endif