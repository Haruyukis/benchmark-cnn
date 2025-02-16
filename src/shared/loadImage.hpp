#ifdef __cplusplus
extern "C" {
#endif

#include <complex.h>



/*
Image Processing: Load the image and put it into a float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    channels: pointer to the number of channels of the image
*/
extern float** loadImageF(const char* chemin_image, int* width, int* height, int* channels);

/*
Image Processing: Load the image and put it into a float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    nb_channels: pointer to the number of channels of the image
*/
extern float* loadImageFptr(const char* chemin_image, int* width, int* height, int* nb_channels);


/*
Image Processing: Load the image and put it into a complex float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    channels: pointer to the number of channels of the image
*/
// complex float** loadImageCF(char* chemin_image, int* width, int* height, int* channels);

/*
Image Processing: free the Image as a float**.
*/
extern void freeImageF(float** image, int channels);

/*
Image Processing: free the Image as a complex float**.
*/
// void freeImageCF(complex float** image, int channels);


extern void afficheImageF(float** image, int width, int height, int channels);

#ifdef __cplusplus
}
#endif