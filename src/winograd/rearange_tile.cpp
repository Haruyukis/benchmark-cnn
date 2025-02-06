#include <iostream>
using namespace std;

/**
 * Reshapes an `n x n` image into `nb_tiles` tiles of `4x4`, stored in row-major order.
 *
 * @param image: Input image stored as a `float*` array of size `n * n`.
 * @param n: The size of the image (assumed to be square: `n x n`).
 * @param output: Output image stored as a `float*` array of size `n * n`.
 */
void reshape_into_patches(const float *image, float *output, int n)
{
    int tile_size = 4;
    int tiling_stride = 2;                                  // Tiling stride to ensure convolution stride of 1
    int tile_per_row = (n - tile_size) / tiling_stride + 1; // Calculate how many tiles per row
    int nb_tiles = tile_per_row * tile_per_row;             // Total number of tiles

    int tile_idx = 0;
    for (int i = 0; i <= n - tile_size; i += tiling_stride) // Iterate over rows
    {
        for (int j = 0; j <= n - tile_size; j += tiling_stride) // Iterate over columns
        {
            // Fill each 4x4 patch
            for (int x = 0; x < tile_size; x++)
            {
                for (int y = 0; y < tile_size; y++)
                {
                    output[tile_idx * tile_size * tile_size + x * tile_size + y] =
                        image[(i + x) * n + (j + y)];
                }
            }
            tile_idx++;
        }
    }
}

/**
 * Prints an array of `float` values in a grid format.
 */
void print_patches(float *tile, int nb_tiles, int tile_size)
{
    for (int i = 0; i < nb_tiles; i++)
    {
        cout << "Tile " << i << ":\n";
        for (int x = 0; x < tile_size; x++)
        {
            for (int y = 0; y < tile_size; y++)
            {
                cout << tile[i * tile_size * tile_size + x * tile_size + y] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    for (int i = 0; i < 144; i++)
    {
        cout << tile[i] << " ";
    }
}
void rearrange_to_2x2_grid_major(float *input_image, float *output_image, int n)
{
    // Each block is 2x2; there are (n/2) blocks per row.
    int block_size = 2;
    int blocks_per_row = n / block_size;
    int num_blocks = blocks_per_row * blocks_per_row; // total number of 2x2 blocks

    int current = 0; // index into input_image (which is in linear order)
    for (int b = 0; b < num_blocks; b++)
    {
        // Determine the block's position in the output grid:
        int r_block = b / blocks_per_row; // block row index (0..blocks_per_row-1)
        int c_block = b % blocks_per_row; // block column index (0..blocks_per_row-1)
        // The starting coordinates (row, col) for this block in the output matrix:
        int base_row = r_block * block_size;
        int base_col = c_block * block_size;

        // Fill the 2x2 block (in row-major order) with the next 4 consecutive input elements:
        output_image[(base_row)*n + (base_col)] = input_image[current++];
        output_image[(base_row)*n + (base_col + 1)] = input_image[current++];
        output_image[(base_row + 1) * n + (base_col)] = input_image[current++];
        output_image[(base_row + 1) * n + (base_col + 1)] = input_image[current++];
    }
}

void print_matrix(float *matrix, int n)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cout << matrix[i * n + j] << "\t";
        }
        cout << endl;
    }
}
