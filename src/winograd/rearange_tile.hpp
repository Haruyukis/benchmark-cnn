/**
 * True input to input tile arrangement
 */
void reshape_into_patches(const float *image, float *output, int n);

/**
 * Output tile arrangement to true output.
 */
void rearrange_to_2x2_grid_major(float *input_image, float *output_image, int n);