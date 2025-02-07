#include <cuda_runtime.h>
#include <cuComplex.h>
#include "fftSharedRow.cuh"
#include "../shared/transpose.cuh"
#include "fftShared.cuh"

/*
Fast Fourier Transform: Do the Fast Fourier Transform of imgDevice (2D FFT).
    img_complexe: image on the host ->> only for tests TO BE REMOVED
    imgDevice: image on the device
    width: width of the image
    height: height of the image
    channels: number of channels of the image
*/
