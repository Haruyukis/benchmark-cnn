#ifndef FFTSHARED_CUH
#define FFTSHARED_CUH

#include <cuda_runtime.h>
#include <cuComplex.h>

/*
Fast Fourier Transform: Do the Fast Fourier Transform of imgDevice (2D FFT).
    img_complexe: image on the host ->> only for tests TO BE REMOVED
    imgDevice: image on the device
    width: width of the image
    height: height of the image
    channels: number of channels of the image
*/
void fftShared(cuFloatComplex** img_complexe, cuFloatComplex* imgDevice, int width, int height, int channels);

#endif // HADAMARD_CUH