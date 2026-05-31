#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define NUM_BINS 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper function to map float32 to [0, 255] bin index
__host__ __device__ __inline__ unsigned char floatToBin(float val) {
    if (val <= 0.0f) return 0;
    if (val >= 1.0f) return 255;
    return (unsigned char)(val * 255.0f);
}

void histogram_cpu(const float* h_data, unsigned int* h_bins, int N) {
    for (int i = 0; i < NUM_BINS; ++i) {
        h_bins[i] = 0;
    }
    for (int i = 0; i < N; ++i) {
        unsigned char bin = floatToBin(h_data[i]);
        h_bins[bin]++;
    }
}

bool verify_results(const unsigned int* reference, const unsigned int* gpu_result) {
    for (int i = 0; i < NUM_BINS; ++i) {
        if (reference[i] != gpu_result[i]) {
            std::cerr << "Mismatch at bin [" << i << "]: CPU=" << reference[i] 
                      << ", GPU=" << gpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

__global__ void histogram_naive(const float* __restrict__ d_data, unsigned int* d_bins, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned char bin = floatToBin(d_data[tid]);
        atomicAdd(&d_bins[bin], 1);
    }
}

__global__ void histogram_shared(const float* __restrict__ d_data, unsigned int* d_bins, int N) {
    __shared__ unsigned int s_bins[NUM_BINS];
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = local_tid; i < NUM_BINS; i += blockDim.x) {
        s_bins[i] = 0;
    }
    __syncthreads();

    if (global_tid < N) {
        unsigned char bin = floatToBin(d_data[global_tid]);
        atomicAdd(&s_bins[bin], 1);
    }
    __syncthreads();

    for (int i = local_tid; i < NUM_BINS; i += blockDim.x) {
        unsigned int val = s_bins[i];
        if (val > 0) {
            atomicAdd(&d_bins[i], val);
        }
    }
}

__global__ void histogram_vectorized(const float4* __restrict__ d_data_v4, unsigned int* d_bins, int N_v4) {
    __shared__ unsigned int s_bins[NUM_BINS];
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = local_tid; i < NUM_BINS; i += blockDim.x) {
        s_bins[i] = 0;
    }
    __syncthreads();

    if (global_tid < N_v4) {
        float4 packed_val = d_data_v4[global_tid];
        atomicAdd(&s_bins[floatToBin(packed_val.x)], 1);
        atomicAdd(&s_bins[floatToBin(packed_val.y)], 1);
        atomicAdd(&s_bins[floatToBin(packed_val.z)], 1);
        atomicAdd(&s_bins[floatToBin(packed_val.w)], 1);
    }
    __syncthreads();

    for (int i = local_tid; i < NUM_BINS; i += blockDim.x) {
        unsigned int val = s_bins[i];
        if (val > 0) {
            atomicAdd(&d_bins[i], val);
        }
    }
}

void run_benchmark(int threads_per_block, int N, const float* d_data, const unsigned int* h_cpu_bins) {
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Testing Block Size (Threads per Block): " << threads_per_block << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    unsigned int* d_bins;
    unsigned int h_gpu_bins[NUM_BINS];
    CUDA_CHECK(cudaMalloc(&d_bins, NUM_BINS * sizeof(unsigned int)));

    cudaEvent_t start, stop;
    float milliseconds = 0;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int grid_size = (N + threads_per_block - 1) / threads_per_block;

    // --- Naive Benchmark ---
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogram_naive<<<grid_size, threads_per_block>>>(d_data, d_bins, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_gpu_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Naive Kernel Time:      " << milliseconds << " ms [" 
              << (verify_results(h_cpu_bins, h_gpu_bins) ? "PASS" : "FAIL") << "]" << std::endl;

    // --- Shared Benchmark ---
    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogram_shared<<<grid_size, threads_per_block>>>(d_data, d_bins, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_gpu_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Shared Kernel Time:     " << milliseconds << " ms [" 
              << (verify_results(h_cpu_bins, h_gpu_bins) ? "PASS" : "FAIL") << "]" << std::endl;

    // --- Vectorized Benchmark ---
    int N_v4 = N / 4; 
    int grid_size_v4 = (N_v4 + threads_per_block - 1) / threads_per_block;
    const float4* d_data_v4 = reinterpret_cast<const float4*>(d_data);

    CUDA_CHECK(cudaMemset(d_bins, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaEventRecord(start));
    histogram_vectorized<<<grid_size_v4, threads_per_block>>>(d_data_v4, d_bins, N_v4);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaMemcpy(h_gpu_bins, d_bins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Vectorized Kernel Time: " << milliseconds << " ms [" 
              << (verify_results(h_cpu_bins, h_gpu_bins) ? "PASS" : "FAIL") << "]" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_bins));
}

int main() {
    const int N = 100000; 
    size_t bytes = N * sizeof(float);

    float* h_data = (float*)malloc(bytes);
    unsigned int h_cpu_bins[NUM_BINS];

    for (int i = 0; i < N; ++i) {
        // h_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        h_data[i] = 100.05;
    }
    histogram_cpu(h_data, h_cpu_bins, N);

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    run_benchmark(128, N, d_data, h_cpu_bins);
    run_benchmark(256, N, d_data, h_cpu_bins);
    run_benchmark(512, N, d_data, h_cpu_bins);
    run_benchmark(1024, N, d_data, h_cpu_bins);

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);
    return 0;
}