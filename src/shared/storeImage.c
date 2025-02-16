#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif
#include <complex.h>
#include "storeImage.hpp"

/*
Image Processing: Convert the image represented as a float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
void storeImageF(const char* path, float** image, int width, int height, int channels){
    unsigned char* output = malloc(height*width*channels*sizeof(unsigned char));
    for (int i =0; i<height; i++){
        for (int j=0; j<width;j++){
            for (int channel=0; channel < channels; channel++){
                output[i*width*channels +j*channels + channel] = (unsigned char) cabsf(image[channel][i*width + j]);
            }
        }
    }
    stbi_write_jpg(path, width, height, channels, output, 90);
    free(output);
}


/*
Image Processing: Convert the image represented as a complex float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
void storeImageCF(char* path, complex float** image, int width, int height, int channels){
    unsigned char* output = malloc(height*width*channels*sizeof(unsigned char));
    for (int i =0; i<height; i++){
        for (int j=0; j<width;j++){
            for (int channel=0; channel < channels; channel++){
                output[i*width*channels +j*channels + channel] = (unsigned char) cabsf(image[channel][i*width + j]);
            }
        }
    }
    stbi_write_jpg(path, width, height, channels, output, 90);
    free(output);
}


void storeImageFptr(const char* path, float* image, int width, int height, int nb_channels){
    unsigned char* output = malloc(height * width * nb_channels * sizeof(unsigned char));
    if (!output) return;  // Handle allocation failure

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int c = 0; c < nb_channels; c++) {
                // Corrected indexing
                output[i * width * nb_channels + j * nb_channels + c] = 
                    (unsigned char) cabsf((image[c * width * height + i * width + j]));  
            }
        }
    }

    stbi_write_jpg(path, width, height, nb_channels, output, 90);
    free(output);
}
