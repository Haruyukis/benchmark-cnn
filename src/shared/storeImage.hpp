#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>

/*
Image Processing: Convert the image represented as a float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
extern void storeImageF(const char* path, float** image, int width, int height, int channels);

/*
Image Processing: Convert the image represented as a complex float** into a jpeg image.
    image: float** containing the image
    path: char* path to save the image
*/ 
// void storeImageCF(char* path, complex float** image, int width, int height, int channels);

/**
 * Image Processing: Convert the image represented as a float * into a jpeg image.
 * image: float* containing the image
 * path: char* path to save the image
 * width: width of the image
 * height: height of the image
 * nb_channels: number of channels of the image
 */
extern void storeImageFptr(const char* path, float* image, int width, int height, int nb_channels);


#ifdef __cplusplus
}
#endif