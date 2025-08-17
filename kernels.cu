#include <cuda_runtime.h>
#include <cublas_v2.h>
#define TILE_WIDTH 16

extern "C" void launch_add_arrays(int *d_a, int *d_b, int *d_c, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
}

extern "C" void launchMatrixMulNoCoalescing(float *d_A, float *d_B, float *d_C, int rowsA, int colsA, int colsB) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((rowsA + TILE_WIDTH - 1) / TILE_WIDTH, (colsB + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMulNoCoalescing<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
}

extern "C" void launchMatrixMulCoalescing(float *d_A, float *d_B, float *d_C, int rowsA, int colsA, int colsB) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulCoalescing<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
}

extern "C" void launchMatrixMulTiled(float *d_A, float *d_B, float *d_C, int rowsA, int colsA, int colsB) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((colsB + TILE_WIDTH - 1) / TILE_WIDTH, (rowsA + TILE_WIDTH - 1) / TILE_WIDTH);
    matrixMulTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
}

// Kernel to add 2 vectors
__global__ void add_arrays(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel Without Memory Coalescing
__global__ void matrixMulNoCoalescing(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = threadIdx.x + blockIdx.x * blockDim.x; 
    int col = threadIdx.y + blockIdx.y * blockDim.y;

    if (row < rowsA && col < colsB) {
        float value = 0;
        for (int k = 0; k < colsA; k++) {
            value += A[row * colsA + k] * B[k * colsB + col]; 
        }
        C[row * colsB + col] = value;
    }
}

// Kernel With Memory Coalescing
__global__ void matrixMulCoalescing(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float value = 0;
        for (int k = 0; k < colsA; k++) {
            value += A[row * colsA + k] * B[k * colsB + col]; 
        }
        C[row * colsB + col] = value;
    }
}

// Tiled Multiplication with Shared Memory
__global__ void matrixMulTiled(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0;

    for (int t = 0; t < (colsA + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < rowsA && t * TILE_WIDTH + threadIdx.x < colsA) {
            sharedA[threadIdx.y][threadIdx.x] = A[row * colsA + t * TILE_WIDTH + threadIdx.x];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (t * TILE_WIDTH + threadIdx.y < colsA && col < colsB) {
            sharedB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * colsB + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = value;
    }
}

