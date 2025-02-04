#include <complex.h>

/*
Image Processing: Load the image and put it into a float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    channels: pointer to the number of channels of the image
*/
float** loadImageF(char* chemin_image, int* width, int* height, int* channels);


/*
Image Processing: Load the image and put it into a complex float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    channels: pointer to the number of channels of the image
*/
complex float** loadImageCF(char* chemin_image, int* width, int* height, int* channels);
