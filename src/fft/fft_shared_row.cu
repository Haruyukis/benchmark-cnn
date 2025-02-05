#include <cuComplex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256

__global__ void fft_DIF_on_rows(cuFloatComplex* image_float, int width, int height, int log2width){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    cuFloatComplex* row_data = &image_float[row*width];

    extern __shared__ cuFloatComplex shared_row[];

    // Load row of the image into shared memory
    if (tid < width){
        shared_row[tid] = row_data[tid];
    }
    __syncthreads();

    // FFT computation (Cooley-Tukey algo, radix 2)
    for (int stage = log2width; stage >= 1; --stage) {
        int m = 1 << stage; // FFT size for this stage
        int m2 = m >> 1; // m / 2
        int j = tid % m;
        int k = tid / m;

        if (j < m2) {
            int idx1 = k * m + j;
            int idx2 = idx1 + m2;

            // Twiddle factor calculation
            float angle = -2.0f * M_PI * j / m;
            cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

            // Butterfly computation
            cuFloatComplex t = shared_row[idx2];
            cuFloatComplex u = shared_row[idx1];

            shared_row[idx1] = cuCaddf(u, t);
            shared_row[idx2] = cuCmulf(w,cuCsubf(u, t));
        }
        __syncthreads();
    }
    // Store the result back to global memory
    if (tid < width) {
        row_data[tid] = shared_row[tid];
    }
}

__global__ void ifft_DIT_on_rows(cuFloatComplex* image_float, int width, int height, int log2width){
    int row = blockIdx.x;
    int tid = threadIdx.x;

    cuFloatComplex* row_data = &image_float[row*width];

    extern __shared__ cuFloatComplex shared_row[];

    // Load row of the image into shared memory
    if (tid < width){
        shared_row[tid] = row_data[tid];
    }
    __syncthreads();

    // FFT computation (Cooley-Tukey algo, radix 2)
    for (int stage = 1; stage <= log2width; ++stage) {
        int m = 1 << stage; // FFT size for this stage
        int m2 = m >> 1; // m / 2
        int j = tid % m;
        int k = tid / m;

        if (j < m2) {
            int idx1 = k * m + j;
            int idx2 = idx1 + m2;

            // Twiddle factor calculation
            float angle = 2.0f * M_PI * j / m;
            cuFloatComplex w = make_cuFloatComplex(cosf(angle), sinf(angle));

            // Butterfly computation
            cuFloatComplex t = cuCmulf(w, shared_row[idx2]);
            cuFloatComplex u = shared_row[idx1];

            shared_row[idx1] = cuCaddf(u, t);
            shared_row[idx2] = cuCsubf(u, t);
        }
        __syncthreads();
    }
    // Store the result back to global memory
    if (tid < width) {
        row_data[tid] = shared_row[tid];
    }
}