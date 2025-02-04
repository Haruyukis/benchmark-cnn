#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
#include "loadImage.hpp"


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
    return imageTensor;
}

float** loadImageF(char* chemin_image, int* width, int* height, int* channels){
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

void freeImageF(float** image, int channels){
    for (int channel =0; channel<channels; channel++){
        free(image[channel]);
    }
    free(image);
}

void freeImageCF(complex float** image, int channels){
    for (int channel =0; channel<channels; channel++){
        free(image[channel]);
    }
    free(image);
}

int main(){
    int width, height, channels;
    char* chemin_image = "../../data/poupoupidou.jpg";
    float** image = loadImageF(chemin_image, &width, &height, &channels);
    afficheImageF(image, width, height, channels);
    freeImageF(image, channels);
}