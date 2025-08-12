#include <cuda_runtime.h>

__global__ void add_arrays(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void launch_add_arrays(int *d_a, int *d_b, int *d_c, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
}