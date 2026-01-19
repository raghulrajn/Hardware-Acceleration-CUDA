#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_fp16.h>

void launch_wmma_fp16_gemm(
    const half* A,
    const half* B,
    float* C,
    int P
);

#endif