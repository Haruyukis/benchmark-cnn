#include "rearange_tile.hpp"

void input_to_tiles(float *input)
{
}

void tiles_to_output(float *output_tiles, float *output, int nb_tiles, int output_width)
{
    for (int tile_idx = 0; tile_idx < nb_tiles; tile_idx++)
    {
        int old_base = tile_idx * 4;
        int new_base = (tile_idx % (output_width / 2)) * 2 + (tile_idx / (output_width / 2)) * 2 * output_width;

        output[new_base] = output_tiles[old_base];
        output[new_base + 1] = output_tiles[old_base + 1];

        output[new_base + output_width] = output_tiles[old_base + 2];
        output[new_base + output_width + 1] = output_tiles[old_base + 3];
    }
}
