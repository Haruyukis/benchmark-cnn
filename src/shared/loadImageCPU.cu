#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <vector>
#include <iostream>

void loadImageCPU(const char* path, std::vector<float>& imgFloat, int* width, int* height, int* channels) {
    unsigned char* imgCharHost = stbi_load(path, width, height, channels, 0);
    if (!imgCharHost) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return;
    }

    int imgSize = (*width) * (*height) * (*channels);
    imgFloat.resize(imgSize);

    for (int i = 0; i < imgSize; i++) {
        imgFloat[i] = static_cast<float>(imgCharHost[i]);
    }

    stbi_image_free(imgCharHost);
}
