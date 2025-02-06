// int main()
// {
//     int n = 12; // working with an 8x8 matrix
//     float *input_image = new float[n * n];
//     float *output_image = new float[n * n];

//     // Initialize the input image with values 0, 1, 2, ..., 63.
//     for (int i = 0; i < n * n; ++i)
//     {
//         input_image[i] = static_cast<float>(i);
//     }

//     cout << "Original Image:" << endl;
//     print_matrix(input_image, n);

//     // Rearrange the input into 2x2 blocks arranged in grid-major order.
//     rearrange_to_2x2_grid_major(input_image, output_image, n);

//     cout << "\nReordered Image into 2x2 Grid-Major Order:" << endl;
//     print_matrix(output_image, n);

//     delete[] input_image;
//     delete[] output_image;

//     return 0;
// }