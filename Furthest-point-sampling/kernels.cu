#include "kernels.h"
#include <cuda_runtime.h>

__global__ void add_kernel(int *a, int *b, int *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void launch_add_kernel(int *a, int *b, int *c, int n) {
    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    add_kernel<<<blocks_per_grid, threads_per_block>>>(a, b, c, n);

    cudaDeviceSynchronize();
}

__device__ void __update(float* __restrict__ dists, int* __restrict__ subs, int tid, int l) {
    if (dists[tid] < dists[l]) {
        dists[tid] = dists[l];
        subs[tid] = subs[l];
    }
}

// __restrict__,  allow the GPU to use its specialized Read-Only Cache, 
// which is much faster than standard global memory for reading the point cloud coordinates

__global__ void farthest_point_sampling_kernel(
    int num_points, 
    int num_samples, 
    const float* __restrict__ point_cloud, 
    float* __restrict__ min_distances, 
    int* __restrict__ sampled_indices
) {

    int local_thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float shared_storage[];
    float* shared_max_dists = shared_storage;
    int* shared_max_indices = (int*)&shared_storage[num_threads];

    int current_seed_idx = 0;
    if (local_thread_id == 0) {
        sampled_indices[0] = current_seed_idx;
    }

    for (int sample_step = 1; sample_step < num_samples; sample_step++) {
        float thread_local_best_dist = -1.0f;
        int thread_local_best_idx = 0;

        float anchor_x = point_cloud[current_seed_idx * 3 + 0];
        float anchor_y = point_cloud[current_seed_idx * 3 + 1];
        float anchor_z = point_cloud[current_seed_idx * 3 + 2];

        //total 512 threads are created and run in parallel
        //To calculate the total 11K points - > this loop run 22 times
        //Each loop operation is strided by num_threads(Block Dimension)
        for (int point_idx = local_thread_id; point_idx < num_points; point_idx += num_threads) {
            float px = point_cloud[point_idx * 3 + 0];
            float py = point_cloud[point_idx * 3 + 1];
            float pz = point_cloud[point_idx * 3 + 2];

            // Squared Euclidean Distance
            float diff_x = px - anchor_x;
            float diff_y = py - anchor_y;
            float diff_z = pz - anchor_z;
            float dist_to_anchor = (diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z);
            
            float current_min = fminf(dist_to_anchor, min_distances[point_idx]);
            min_distances[point_idx] = current_min;

            if (current_min > thread_local_best_dist) {
                thread_local_best_dist = current_min;
                thread_local_best_idx = point_idx;
            }
        }

        shared_max_dists[local_thread_id] = thread_local_best_dist;
        shared_max_indices[local_thread_id] = thread_local_best_idx;
        __syncthreads();

        for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
            if (local_thread_id < stride) {

                if (shared_max_dists[local_thread_id] < shared_max_dists[local_thread_id + stride]) {
                    shared_max_dists[local_thread_id] = shared_max_dists[local_thread_id + stride];
                    shared_max_indices[local_thread_id] = shared_max_indices[local_thread_id + stride];
                }
            }
            __syncthreads();
        }

        current_seed_idx = shared_max_indices[0];
        if (local_thread_id == 0) {
            sampled_indices[sample_step] = current_seed_idx;
        }
        __syncthreads();
    }
}


__global__ void farthest_point_sampling_batched_kernel(
    int num_points, 
    int num_samples, 
    const float* __restrict__ point_cloud, 
    float* __restrict__ min_distances, 
    int* __restrict__ sampled_indices
) {
    int batch_idx = blockIdx.x; 
    int local_thread_id = threadIdx.x;
    int num_threads = blockDim.x;

    // Point cloud is [B, N, 3], distances are [B, N], indices are [B, M]
    point_cloud += batch_idx * num_points * 3;
    min_distances += batch_idx * num_points;
    sampled_indices += batch_idx * num_samples;

    extern __shared__ float shared_storage[];
    float* shared_max_dists = shared_storage;
    int* shared_max_indices = (int*)&shared_storage[num_threads];

    int current_seed_idx = 0;
    if (local_thread_id == 0) {
        sampled_indices[0] = current_seed_idx;
    }

    for (int sample_step = 1; sample_step < num_samples; sample_step++) {
        float thread_local_best_dist = -1.0f;
        int thread_local_best_idx = 0;

        float anchor_x = point_cloud[current_seed_idx * 3 + 0];
        float anchor_y = point_cloud[current_seed_idx * 3 + 1];
        float anchor_z = point_cloud[current_seed_idx * 3 + 2];

        for (int point_idx = local_thread_id; point_idx < num_points; point_idx += num_threads) {
            float px = point_cloud[point_idx * 3 + 0];
            float py = point_cloud[point_idx * 3 + 1];
            float pz = point_cloud[point_idx * 3 + 2];

            float diff_x = px - anchor_x;
            float diff_y = py - anchor_y;
            float diff_z = pz - anchor_z;
            float dist_to_anchor = (diff_x * diff_x) + (diff_y * diff_y) + (diff_z * diff_z);
            
            // This is where the "distance to group" is stored and updated
            float current_min = fminf(dist_to_anchor, min_distances[point_idx]);
            min_distances[point_idx] = current_min;

            if (current_min > thread_local_best_dist) {
                thread_local_best_dist = current_min;
                thread_local_best_idx = point_idx;
            }
        }

        shared_max_dists[local_thread_id] = thread_local_best_dist;
        shared_max_indices[local_thread_id] = thread_local_best_idx;
        __syncthreads();

        for (int stride = num_threads / 2; stride > 0; stride >>= 1) {
            if (local_thread_id < stride) {
                if (shared_max_dists[local_thread_id] < shared_max_dists[local_thread_id + stride]) {
                    shared_max_dists[local_thread_id] = shared_max_dists[local_thread_id + stride];
                    shared_max_indices[local_thread_id] = shared_max_indices[local_thread_id + stride];
                }
            }
            __syncthreads();
        }

        current_seed_idx = shared_max_indices[0];
        if (local_thread_id == 0) {
            sampled_indices[sample_step] = current_seed_idx;
        }
        __syncthreads();
    }
}
void launch_fps(int n, int m, const float* dataset, float* temp_dist, int* idxs) {
    int threads = 1024;
    size_t shared_size = threads * (sizeof(float) + sizeof(int));
    farthest_point_sampling_kernel<<<1, threads, shared_size>>>(n, m, dataset, temp_dist, idxs);
}

void launch_fps_batched(int b, int n, int m, const float* dataset, float* temp_dist, int* idxs) {
    int threads = 512;
    size_t shared_size = threads * (sizeof(float) + sizeof(int));
    farthest_point_sampling_batched_kernel<<<b, threads, shared_size>>>(n, m, dataset, temp_dist, idxs);
}