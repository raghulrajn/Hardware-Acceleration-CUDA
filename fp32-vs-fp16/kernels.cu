#include <cuda_runtime.h>
#include <mma.h>
#include "kernels.h"

using namespace nvcuda;

__global__ void wmma_fp16_gemm_kernel(
    const half* A,
    const half* B,
    float* C,
    int P
) {
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < P; k += 16) {
        const half* tileA = A + (warpM * 16) * P + k;
        const half* tileB = B + k * P + (warpN * 16);

        wmma::load_matrix_sync(a_frag, tileA, P);
        wmma::load_matrix_sync(b_frag, tileB, P);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* tileC = C + (warpM * 16) * P + (warpN * 16);
    wmma::store_matrix_sync(tileC, c_frag, P, wmma::mem_row_major);
}

void launch_wmma_fp16_gemm(
    const half* A,
    const half* B,
    float* C,
    int P
) {
    dim3 blocks(P / 16, P / 16);
    dim3 threads(32, 1, 1);  // one warp

    wmma_fp16_gemm_kernel<<<blocks, threads>>>(A, B, C, P);
}
