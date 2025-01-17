__global__ void
transpose(float* A, float* T, int heightA, int widthA) // A: source matrix; T: matrix to be transposed
{
  int Ax = blockDim.x * blockIdx.x + threadIdx.x;
  int Ay = blockDim.y * blockIdx.y + threadIdx.y;

  if (Ax < widthA && Ay < heightA) {

    int indexA = Ax + widthA * Ay;

    int blockDimY = blockDim.y + 1;
    extern __shared__ float tile[]; // blockDimY = blockDim.y + 1 to avoid bank conflicts

    int indexTile = threadIdx.y + blockDimY * threadIdx.x;

    tile[indexTile] = A[indexA]; // Transposition

    __syncthreads();

    indexTile = threadIdx.x + blockDimY * threadIdx.y;
    int indexT = Ay + heightA * Ax;
    T[indexT] = tile[indexTile]; // Write to destination matrix
  }
  else {
    printf("Index x or y out of bounds: x = %i, y = %i\n", Ax, Ay);
  }
}