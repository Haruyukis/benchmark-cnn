// Déclarer les premiers &arguments avant d'appeler cette fonction
void get_args_transpose(dim3 &grid, dim3 &threads, unsigned int &shared, int block_size, int widthA, int heightA) {
  threads.x = block_size;
  threads.y = block_size;
  grid.x = std::ceil((float)widthA / (float)threads.x);
  grid.y = std::ceil((float)heightA / (float)threads.y);
  shared = block_size * (block_size+1) * sizeof(float);
}

__global__ void transpose(float* A, float* T, int heightA, int widthA) {
    int Ax = blockDim.x * blockIdx.x + threadIdx.x;
    int Ay = blockDim.y * blockIdx.y + threadIdx.y;

    int blockDimY = blockDim.y + 1;
    extern __shared__ float tile[];

    int indexTile = threadIdx.x + blockDimY * threadIdx.y;

    // Load data into shared memory, handling out-of-bounds threads
    if (Ax < widthA && Ay < heightA) {
        int indexA = Ax + widthA * Ay;
        tile[indexTile] = A[indexA];
    } else {
        tile[indexTile] = 0.0f; // Optional, as writes to T are guarded
    }

    __syncthreads(); // All threads participate

    // Calculate transposed indices and write to T if in bounds
    indexTile = threadIdx.x + blockDimY * threadIdx.y;
    int indexT = Ay + heightA * Ax;

    if (Ax < widthA && Ay < heightA) {
        T[indexT] = tile[indexTile];
    }
}