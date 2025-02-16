#include "winograd/winograd.hpp"
#include "shared/loadImage.hpp"
#include "shared/storeImage.hpp"
#include "shared/utils.hpp"
#include <iostream>
#include <chrono>

auto startTimer() {
    return std::chrono::high_resolution_clock::now();
}

void stopTimer(const std::string &label, const std::chrono::high_resolution_clock::time_point &start) {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << label << " took " << duration.count() << " seconds." << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <chemin_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path = argv[1];
    int width, height, nb_channels;
    auto startLoad = startTimer();
    float *input = loadImageFptr(path, &width, &height, &nb_channels);
    stopTimer("loadImageFptr", startLoad);

    int o_width = width - 2;
    int o_height = height - 2;
    int i_size = width * height;
    int o_size = o_width * o_height;
    float *output = new float[o_size * nb_channels];
    float *filter = new float[9]{
        -0.23243f, -1.23f, -2.5342423f,
        3.0f, -5.64f, 0.3232f,
        0.0034342f, -.34256f, 0.3451f};
    auto startWinograd = startTimer();
    winograd_cpu(output, input, filter, width, height, 3, 3, nb_channels);
    stopTimer("winograd_cpu", startWinograd);

    auto startStore = startTimer();
    storeImageFptr("output_cpu.jpg", output, o_width, o_height, nb_channels);
    stopTimer("storeImageFptr", startStore);

    delete[] output;
    delete[] filter;
    delete[] input;

    return 0;
}