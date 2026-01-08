#include <iostream>
#include <cuda_runtime.h>
#include "kernels.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

void fps_cpu(int n, int m, const float* points, int* idxs) {
    std::vector<float> min_dists(n, 1e10f);
    int last_idx = 0;
    idxs[0] = last_idx;

    for (int j = 1; j < m; j++) {
        float max_min_dist = -1.0f;
        int next_idx = 0;

        float x1 = points[last_idx * 3 + 0];
        float y1 = points[last_idx * 3 + 1];
        float z1 = points[last_idx * 3 + 2];

        for (int i = 0; i < n; i++) {
            float x2 = points[i * 3 + 0];
            float y2 = points[i * 3 + 1];
            float z2 = points[i * 3 + 2];
            
            float d2 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
            if (d2 < min_dists[i]) min_dists[i] = d2;

            if (min_dists[i] > max_min_dist) {
                max_min_dist = min_dists[i];
                next_idx = i;
            }
        }
        last_idx = next_idx;
        idxs[j] = last_idx;
    }
}
int main() {
   
    int N = 11236; // Input points
    int M = 5000;    // Points to sample
    
    std::vector<float> h_points(N * 3);
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0, 100.0);
    
    for(int i = 0; i < N * 3; i++) h_points[i] = dis(gen);

    float *d_points, *d_temp_dist;
    int *d_idxs;
    std::vector<int> cpu_idxs(M),gpu_idxs(M);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    fps_cpu(N, M, h_points.data(), cpu_idxs.data());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = end_cpu - start_cpu;

    cudaMalloc(&d_points, N * 3 * sizeof(float));
    cudaMalloc(&d_temp_dist, N * sizeof(float));
    cudaMalloc(&d_idxs, M * sizeof(int));

    std::vector<float> h_temp_dist(N, 1e10f);

    cudaMemcpy(d_points, h_points.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_dist, h_temp_dist.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    auto start_gpu = std::chrono::high_resolution_clock::now();
    launch_fps(N, M, d_points, d_temp_dist, d_idxs);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_ms = end_gpu - start_gpu;

    cudaMemcpy(gpu_idxs.data(), d_idxs, M * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "CPU Time for 1 point cloud: " << cpu_ms.count() << " ms" << std::endl;
    std::cout << "GPU Time for 1 Point cloud: " << gpu_ms.count() << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_ms.count() / gpu_ms.count() << "x" << std::endl;

    // bool match = true;
    // for(int i=0; i<5; i++) if(cpu_idxs[i] != gpu_idxs[i]) match = false;
    // std::cout << "Verification (First 5 indices match): " << (match ? "YES" : "NO") << std::endl;

    // for(int i=0; i<5; i++) {
    //     std::cout << "CPU idx: " << cpu_idxs[i] << ", GPU idx: " << gpu_idxs[i] << std::endl;}
    // cudaFree(d_points); cudaFree(d_temp_dist); cudaFree(d_idxs);

    
    {int B = 10;    // 10 point clouds
    int N = 11000; // 11k points each
    int M = 5000;   // 100 samples each

    // Host memory
    std::vector<float> h_dataset(B * N * 3);
    std::vector<float> h_init_dist(B * N, 1e10f); // Initialize to infinity
    std::vector<int> h_idxs(B * M);

    std::vector<float> h_points(N * 3);
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0, 100.0);
    
for(int i = 0; i < B * N * 3; i++) {
    h_dataset[i] = dis(gen);
}

    // float *d_points, *d_temp_dist;
    // int *d_idxs;
    // std::vector<int> cpu_idxs(M),gpu_idxs(M);


    // Device memory
    float *d_dataset, *d_temp_dist;
    int *d_idxs;
    cudaMalloc(&d_dataset, B * N * 3 * sizeof(float));
    cudaMalloc(&d_temp_dist, B * N * sizeof(float));
    cudaMalloc(&d_idxs, B * M * sizeof(int));

    // Copy to device
    cudaMemcpy(d_dataset, h_dataset.data(), B * N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_dist, h_init_dist.data(), B * N * sizeof(float), cudaMemcpyHostToDevice);

    auto start_gpu_batch = std::chrono::high_resolution_clock::now();
    launch_fps_batched(B, N, M, d_dataset, d_temp_dist, d_idxs);
    cudaDeviceSynchronize();
    auto end_gpu_batch = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_ms_batch = end_gpu_batch - start_gpu_batch;
    std::cout << "GPU Time for batch of 10 point clouds: " << gpu_ms_batch.count() << " ms" << std::endl;

    cudaMemcpy(h_idxs.data(), d_idxs, B * M * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dataset); cudaFree(d_temp_dist); cudaFree(d_idxs);}
    return 0;

}