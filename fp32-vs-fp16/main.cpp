#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cublas_v2.h>

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

    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
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

    std::cout << "H2D copy time:  " << time_h2d  << " ms\n";
    std::cout << "GEMM time:     " << time_gemm << " ms\n";
    std::cout << "D2H copy time: " << time_d2h  << " ms\n";
    std::cout << "Total time:    " 
              << (time_h2d + time_gemm + time_d2h) << " ms\n";

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

    return 0;
}
