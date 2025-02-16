#include "winograd/winograd.hpp"
#include "shared/loadImage.hpp"
#include "shared/storeImage.hpp"
#include "shared/utils.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <chemin_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path = argv[1];
    int width, height, nb_channels;
    float *input = loadImageFptr(path, &width, &height, &nb_channels);

    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *output = new float[o_size * nb_channels];
    float *filter = new float[9]{
        -1.0f, -1.f, -1.f,
        0.0f, 0.f, 0.f,
        1.0f, 1.f, 1.f};

    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channels);
    storeImageFptr("output_cpu.jpg", output, o_width, o_height, nb_channels);

    return 0;
}