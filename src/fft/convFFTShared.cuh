#include <cuComplex.h>

/* Convolution by using FFT transform of imgDevice and kernelDevice
    img_complexe: image on the host ->> only for tests TO BE REMOVED
    imgDevice: image on the device
    kernelDevice: kernel on the device
    width: width of the image
    height: height of the image
    channels: number of channels of the image*/
void convFFTShared(cuFloatComplex* imgDevice, cuFloatComplex* kernelDevice, int width, int height, int channels);