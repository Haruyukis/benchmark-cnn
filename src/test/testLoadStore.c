#include "loadImage.hpp"
#include "storeImage.hpp"
#include <complex.h>

int main(){
    int width, height, channels;
    char* chemin_image = "../../data/poupoupidou.jpg";
    complex float** image = loadImageCF(chemin_image, &width, &height, &channels);
    // afficheImageCF(image, width, height, channels);
    char* chemin_image_sortie = "../../data/test store.jpg";
    storeImageCF(chemin_image_sortie, image, width, height, channels);
    freeImageCF(image, channels);
    return 0;
}