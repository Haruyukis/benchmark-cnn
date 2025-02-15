#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>
#include <iostream>

void storeImageCPU(std::vector<float>& imgFloat, const char* path, int trueWidth, int trueHeight, int channels) {
    std::vector<unsigned char> imgChar(trueWidth * trueHeight * channels);

    for (int i = 0; i < trueWidth * trueHeight * channels; i++) {
        imgChar[i] = static_cast<unsigned char>(imgFloat[i]);
    }

    if (!stbi_write_jpg(path, trueWidth, trueHeight, channels, imgChar.data(), 90)) {
        std::cerr << "Failed to write image: " << path << std::endl;
    }
}
