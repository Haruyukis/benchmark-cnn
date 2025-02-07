#include <stdint.h>

#ifndef DATA_PROCESS_CUH
#define DATA_PROCESS_CUH
/**
 * True input to input tile arrangement
 */
__global__ void reshape_into_tiles(const float *image, float *output, int n);

/**
 * Output tile arrangement to true output.
 */
void rearrange_to_2x2_grid_major(float *input_image, float *output_image, int n);
#endif // DATA_PROCESS_CUH