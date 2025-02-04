#include <complex.h>

/*
Image Processing: Convert the image represented as a float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
void storeImageF(char* path, float** image, int width, int height, int channels);

/*
Image Processing: Convert the image represented as a complex float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
void storeImageCF(char* path, complex float** image, int width, int height, int channels);