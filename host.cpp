#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define TILE_WIDTH 16
extern "C" void launch_add_arrays(int *d_a, int *d_b, int *d_c, int n);
extern "C" void launchMatrixMulNoCoalescing(float *d_A, float *d_B, float *d_C, int rowsA, int colsA, int colsB);
extern "C" void launchMatrixMulCoalescing(float *d_A, float *d_B, float *d_C, int rowsA, int colsA, int colsB);
extern "C" void launchMatrixMulTiled(float *d_A, float *d_B, float *d_C, int rowsA, int colsA, int colsB) ;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void cpuMatrixMul(float *A, float *B, float *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            float value = 0;
            for (int k = 0; k < colsA; k++) {
                value += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = value;
        }
    }
}

int main() {
    {
    const int N = 1000;
    const int SIZE = N * sizeof(int);

    int *h_a = (int *)malloc(SIZE);
    int *h_b = (int *)malloc(SIZE);
    int *h_c = (int *)malloc(SIZE);

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, SIZE);
    cudaMalloc((void **)&d_b, SIZE);
    cudaMalloc((void **)&d_c, SIZE);

    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

    launch_add_arrays(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Array addition successful!" << std::endl;
    } else {
        std::cout << "Error in array addition!" << std::endl;
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

{
    int rowsA = 512, colsA = 512, rowsB = 512, colsB = 512;
    if (colsA != rowsB) {
        std::cerr << "Error: Invalid matrix dimensions for multiplication. "
                  << "Matrix A (" << rowsA << "x" << colsA << ") "
                  << "and Matrix B (" << rowsB << "x" << colsB << ") "
                  << "cannot be multiplied." << std::endl;
        return -1;
    }

    int sizeA = rowsA * colsA * sizeof(float);
    int sizeB = rowsB * colsB * sizeof(float);
    int sizeC = rowsA * colsB * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C1 = (float*)malloc(sizeC);
    float *h_C2 = (float*)malloc(sizeC);
    float *h_C3 = (float*)malloc(sizeC);
    float *h_C_verify = (float*)malloc(sizeC);

    for (int i = 0; i < rowsA * colsA; i++) h_A[i] = rand() % 100 / 100.0f;
    for (int i = 0; i < rowsB * colsB; i++) h_B[i] = rand() % 100 / 100.0f;

    float *d_A, *d_B, *d_C1, *d_C2, *d_C3;
    checkCuda(cudaMalloc((void**)&d_A, sizeA));
    checkCuda(cudaMalloc((void**)&d_B, sizeB));
    checkCuda(cudaMalloc((void**)&d_C1, sizeC));
    checkCuda(cudaMalloc((void**)&d_C2, sizeC));
    checkCuda(cudaMalloc((void**)&d_C3, sizeC));

    checkCuda(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    std::cout<<"Matrix Multiplication w/o Memory Coalescing"<<std::endl;
    launchMatrixMulNoCoalescing(d_A, d_B, d_C1, rowsA, colsA, colsB);
    checkCuda(cudaMemcpy(h_C1, d_C1, sizeC, cudaMemcpyDeviceToHost));
    cpuMatrixMul(h_A, h_B, h_C_verify, rowsA, colsA, colsB);
    
    for (int i = 0; i < rowsA * colsB; i++) {
        if (fabs(h_C1[i] - h_C_verify[i]) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_C1[i] << ", CPU=" << h_C_verify[i] << std::endl;
            return -1;
        }
    }

    std::cout<<"Matrix Multiplication with Memory Coalescing"<<std::endl;
    launchMatrixMulCoalescing(d_A, d_B, d_C2, rowsA, colsA, colsB);
    checkCuda(cudaMemcpy(h_C2, d_C2, sizeC, cudaMemcpyDeviceToHost));
    for (int i = 0; i < rowsA * colsB; i++) {
        if (fabs(h_C2[i] - h_C_verify[i]) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_C3[i] << ", CPU=" << h_C_verify[i] << std::endl;
            return -1;
        }
    }
    std::cout<<"Matrix Multiplication with Tiled Memory"<<std::endl;
    launchMatrixMulTiled(d_A, d_B, d_C3, rowsA, colsA, colsB);
    checkCuda(cudaMemcpy(h_C3, d_C3, sizeC, cudaMemcpyDeviceToHost));
    for (int i = 0; i < rowsA * colsB; i++) {
        if (fabs(h_C3[i] - h_C_verify[i]) > 1e-3) {
            std::cerr << "Mismatch at index " << i << ": GPU=" << h_C3[i] << ", CPU=" << h_C_verify[i] << std::endl;
            return -1;
        }
    }

    std::cout << "Matrix multiplication verified successfully!" << std::endl;


    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    free(h_C3);
    free(h_C_verify);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);

    return 0;
}
}
