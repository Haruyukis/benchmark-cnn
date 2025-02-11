#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef IFFTSHARED_CUH
#define IFFTSHARED_CUH

void ifftShared(cuFloatComplex* imgDevice, int width, int height, int channels);

#endif