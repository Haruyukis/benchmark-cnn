#include <cmath>
#include <iostream>

#ifndef UTILS_HPP
#define UTILS_HPP

bool compare_arrays(const float* A, const float* B, int n, float tolerance);

void printTile(float* input, int startRow, int startCol, int stride, int size, int width);

void printMatrix(float* input, int width, int height);

#endif