#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cublas_v2.h>
#include "kernels.h"

int main() {

    int N = 1000;
    size_t bytes = N * N * sizeof(float);

    float *d_A32, *d_B32, *d_C32;
    cudaMalloc(&d_A32, bytes);
    cudaMalloc(&d_B32, bytes);
    cudaMalloc(&d_C32, bytes);

    std::vector<float> h_A32(N * N), h_B32(N * N), h_C32(N * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < N * N; i++) {
        h_A32[i] = dis(gen);
        h_B32[i] = dis(gen);
    }

    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t gemm_start, gemm_stop;
    cudaEvent_t d2h_start, d2h_stop;

    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&gemm_start);
    cudaEventCreate(&gemm_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);


    cudaEventRecord(h2d_start);

    cudaMemcpy(d_A32, h_A32.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B32, h_B32.data(), bytes, cudaMemcpyHostToDevice);

    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);

    float time_h2d;
    cudaEventElapsedTime(&time_h2d, h2d_start, h2d_stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    cudaEventRecord(gemm_start);
    // C(m×n) = alpha · A(m×k) · B(k×n) + beta · C(m×n)
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N, //m, n, k
        &alpha,
        d_A32, N,
        d_B32, N,
        &beta,
        d_C32, N
    );

    cudaEventRecord(gemm_stop);
    cudaEventSynchronize(gemm_stop);

    float time_gemm;
    cudaEventElapsedTime(&time_gemm, gemm_start, gemm_stop);

    cudaEventRecord(d2h_start);

    cudaMemcpy(h_C32.data(), d_C32, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(d2h_stop);
    cudaEventSynchronize(d2h_stop);

    float time_d2h;
    cudaEventElapsedTime(&time_d2h, d2h_start, d2h_stop);

    // std::cout << "H2D copy time:  " << time_h2d  << " ms\n";
    // std::cout << "GEMM time:     " << time_gemm << " ms\n";
    // std::cout << "D2H copy time: " << time_d2h  << " ms\n";
    // std::cout << "Total time:    " 
    //           << (time_h2d + time_gemm + time_d2h) << " ms\n";

    cublasDestroy(handle);
    cudaFree(d_A32);
    cudaFree(d_B32);
    cudaFree(d_C32);

    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(gemm_start);
    cudaEventDestroy(gemm_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    half *d_A16_, *d_B16_;
    float *d_C32_;
    
    std::vector<half> h_A16_(N * N );
    std::vector<half> h_B16_(N * N );
    for (int i = 0; i < N * N; i++) {
    h_A16_[i] = __float2half(h_A32[i]);
    h_B16_[i] = __float2half(h_B32[i]);
    }

    cudaMalloc(&d_A16_, N * N * sizeof(half));
    cudaMalloc(&d_B16_, N * N *sizeof(half));
    cudaMalloc(&d_C32_, N * N * sizeof(float));
    cudaMemset(d_C32_, 0, N * N * sizeof(float));


    half alpha16 = 1.0, beta16 = 0.0;
    cublasHandle_t handle16;
    cublasCreate(&handle16);
    int lda = N; //loading dimensions
    int ldb = N;
    int ldc = N;
    cudaEvent_t gemm16_start, gemm16_stop;
    cudaEventCreate(&gemm16_start);
    cudaEventCreate(&gemm16_stop);
    cudaEventRecord(gemm16_start);
    cublasGemmEx(
            handle16,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha16,
            d_A16_,
            CUDA_R_16F,
            lda,
            d_B16_,
            CUDA_R_16F,
            ldb,
            &beta16,
            d_C32_,
            CUDA_R_32F,
            ldc,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
);
    cudaEventRecord(gemm16_stop);
    cudaEventSynchronize(gemm16_stop);

    float time_gemm16;
    cudaEventElapsedTime(&time_gemm16, gemm16_start, gemm16_stop);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // =========================
    // FP16 WMMA GEMM
    // =========================
    int P = ((N + 15) / 16) * 16;   // 1008

    half *d_A16, *d_B16;
    float *d_C16;

    cudaMalloc(&d_A16, P * P * sizeof(half));
    cudaMalloc(&d_B16, P * P * sizeof(half));
    cudaMalloc(&d_C16, P * P * sizeof(float));
    cudaMemset(d_C16, 0, P * P * sizeof(float));

    std::vector<half> h_A16(P * P);
    std::vector<half> h_B16(P * P);

    for (int r = 0; r < P; r++) {
        for (int c = 0; c < P; c++) {
            if (r < N && c < N) {
                h_A16[r * P + c] = __float2half(h_A32[r * N + c]);
                h_B16[c * P + r] = __float2half(h_B32[r * N + c]);
            } else {
                h_A16[r * P + c] = __float2half(0.0f);
                h_B16[c * P + r] = __float2half(0.0f);
            }
        }
    }

    cudaMemcpy(d_A16, h_A16.data(), P * P * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B16, h_B16.data(), P * P * sizeof(half), cudaMemcpyHostToDevice);

    cudaEvent_t start16, stop16;
    cudaEventCreate(&start16);
    cudaEventCreate(&stop16);

    cudaEventRecord(start16);
    launch_wmma_fp16_gemm(d_A16, d_B16, d_C16, P);

    cudaEventRecord(stop16);
    cudaEventSynchronize(stop16);

    float time_fp16;
    cudaEventElapsedTime(&time_fp16, start16, stop16);
    std::cout << "FP32 cuBLAS GEMM compute time: " << time_gemm << " ms\n";
    std::cout << "FP16 cuBLAS GEMM compute time:  " << time_gemm16 << " ms\n";
    std::cout << "FP16 WMMA GEMM compute time:  " << time_fp16 << " ms\n";
    std::cout << "Speedup (FP32 / FP16):        "
            << time_gemm / time_fp16 << "x\n";
    cudaFree(d_A16);
    cudaFree(d_B16);
    cudaFree(d_C16);

    cudaEventDestroy(start16);
    cudaEventDestroy(stop16);


    return 0;
}
