#include <iostream>
#include <cuda_runtime.h>

extern "C" void launch_add_arrays(int *d_a, int *d_b, int *d_c, int n);

int main() {
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

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}