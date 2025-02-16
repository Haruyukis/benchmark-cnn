#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "loadImage.hpp"


/*
Image Processing: Load the image and put it into a float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    channels: pointer to the number of channels of the image
*/
float** loadImageF(const char* chemin_image, int* width, int* height, int* channels){
    unsigned char* img = stbi_load(chemin_image, width, height, channels,0);
    printf("Image chargée, width:%d, height:%d, channels:%d\n",*width, *height, *channels);
    float** imageTensor = malloc(*channels*sizeof(float*));
    for (int channel=0; channel<*channels; channel++){
        imageTensor[channel] = malloc((*width)*(*height)*sizeof(float));
    }
    for (int n = 0; n< (*width)*(*height)*(*channels); n++){
        int channel = n%(*channels);
        int numPixel = n/(*channels);
        imageTensor[channel][numPixel] = (float) img[n];
    }
    stbi_image_free(img);
    return imageTensor;
}

float* loadImageFptr(const char* chemin_image, int* width, int* height, int* nb_channels){
    unsigned char* img = stbi_load(chemin_image, width, height, nb_channels, 0);
    printf("Image chargée, width:%d, height:%d, nb_channels:%d\n",*width, *height, *nb_channels);
    int totalPixels = (*width) * (*height);
    int channels = *nb_channels;

    float* imageTensor = malloc(totalPixels * channels * sizeof(float));
    for (int n = 0; n < totalPixels * channels; n++) {
        int channel = n % channels;
        int numPixel = n / channels;
        imageTensor[channel * totalPixels + numPixel] = (float) img[n];
    }
    stbi_image_free(img);
    return imageTensor;
}



/*
Image Processing: Load the image and put it into a complex float**.
    chemin_image: path to the image
    width: pointer to the width of the image
    height: pointer to the height of the image
    channels: pointer to the number of channels of the image
*/
complex float** loadImageCF(char* chemin_image, int* width, int* height, int* channels){
    unsigned char* img = stbi_load(chemin_image, width, height, channels,0);
    printf("Image chargée, width:%d, height:%d, channels:%d\n",*width, *height, *channels);
    complex float** imageTensor = malloc(*channels*sizeof(complex float*));
    for (int channel=0; channel<*channels; channel++){
        imageTensor[channel] = malloc((*width)*(*height)*sizeof(complex float));
    }
    for (int n = 0; n< (*width)*(*height)*(*channels); n++){
        int channel = n%(*channels);
        int numPixel = n/(*channels);
        imageTensor[channel][numPixel] = (complex float) img[n];
    }
    stbi_image_free(img);
    return imageTensor;
}

/*
Image Processing: free the Image as a float**.
*/
void freeImageF(float** image, int channels){
    for (int channel =0; channel<channels; channel++){
        free(image[channel]);
    }
    free(image);
}

/*
Image Processing: free the Image as a complex float**.
*/
void freeImageCF(complex float** image, int channels){
    for (int channel =0; channel<channels; channel++){
        free(image[channel]);
    }
    free(image);
}


void afficheImageCF(complex float** image, int width, int height, int channels){
    for (int channel = 0; channel<channels; channel++){
        for (int i = 0; i<height;i++){
            for (int j = 0; j<width; j++){
                printf("%03.0f ",crealf(image[channel][j + i*width]));
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void afficheImageF(float** image, int width, int height, int channels){
    for (int channel = 0; channel<channels; channel++){
        for (int i = 0; i<height;i++){
            for (int j = 0; j<width; j++){
                printf("%03.0f ",crealf(image[channel][j + i*width]));
            }
            printf("\n");
        }
        printf("\n\n");
    }
}
/*
int main(){
    int width, height, channels;
    char* chemin_image = "../../data/poupoupidou.jpg";
    float** image = loadImageF(chemin_image, &width, &height, &channels);
    afficheImageF(image, width, height, channels);
    freeImageF(image, channels);
}
*/