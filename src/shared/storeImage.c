#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <complex.h>
#include "storeImage.hpp"

/*
Image Processing: Convert the image represented as a float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
void storeImageF(char* path, float** image, int width, int height, int channels){
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

