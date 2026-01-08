#ifndef KERNELS_H
#define KERNELS_H

void launch_add_kernel(int *a, int *b, int *c, int n);
void launch_fps(int n, int m, const float* dataset, float* temp_dist, int* idxs);
void launch_fps_batched(int b, int n, int m, const float* dataset, float* temp_dist, int* idxs);
#endif