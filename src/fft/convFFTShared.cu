#include <cuComplex.h>
#include "fftShared.cuh"
#include "../shared/hadamard.cuh"
#include "ifftShared.cuh"

// supprimer 
#include "../shared/loadImage.hpp"
#include "../shared/storeImage.hpp"
#include <stdlib.h>

/* Convolution by using FFT transform of imgDevice and kernelDevice
    img_complexe: image on the host ->> only for tests TO BE REMOVED
    imgDevice: image on the device
    kernelDevice: kernel on the device
    width: width of the image
    height: height of the image
    channels: number of channels of the image*/
void convFFTShared(cuFloatComplex* imgDevice, cuFloatComplex* kernelDevice, int width, int height, int channels){
    fftShared(imgDevice, width, height, channels);
    fftShared(kernelDevice, width, height, 1);

    // A supprimer  //  //  //  //
    // int N = width*height;
    // float** image = (float**) malloc(channels*sizeof(float*));
    // for (int channel=0; channel<channels; channel++){
    //     image[channel] = (float*) malloc((width)*(height)*sizeof(float));
    // }
    // for (int channel = 0; channel<channels; channel++){
    //     for (int i = 0; i < N; ++i) {
    //     image[channel][i] = cuCrealf(img_complexe[channel][i]);
    //     // printf("Output[%d] = (%.2f, %.2f)\n", i, cuCrealf(h_input[i]), cuCimagf(h_input[i]));
    //     }
    // }
    // const char* chemin_sortie_inv = "./data/test fft vraiment.jpeg";
    // storeImageF(chemin_sortie_inv, image, width, height, channels);
    // //   //  //  //  //  //  //



    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = blockDim.x * (blockDim.y + 1) * sizeof(cuFloatComplex);

    for (int channel = 0; channel < channels; channel++){
        cuFloatComplex* ptrChannel = imgDevice + channel * width * height;
        hadamard_kernel_Cufloatc<<<gridDim, blockDim, sharedMemSize>>>(ptrChannel, kernelDevice, width, height);
        cudaDeviceSynchronize();
    }
    ifftShared(imgDevice, width, height, channels);
}