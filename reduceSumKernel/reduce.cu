#include <iostream>

// (Warp Divergence) Threads are executed in groups of 32 called warps. 
// All 32 threads in a warp must execute the exact same instruction at the same time. 
// If thread 0 and thread 1 want to do different things, the GPU has to pause one while the other runs.

// Shared Memory Banks: Shared memory is split into 32 distinct memory modules called banks.
// If multiple threads in a warp try to access different addresses within the same bank at the same time, 
// the hardware serializes the requests (Bank Conflict), slowing things down.

__global__ void reduced_sum_naive(float* input, float* output, int n){

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int localId = threadIdx.x;

    extern __shared__ float sharedMem[];

    if(id<n){
        sharedMem[localId] = input[id];
    }
    else{
        sharedMem[localId] = 0;
    }

    __syncthreads();

    // thread 0 passes the if statement, thread 1 fails it, thread 2 passes, thread 3 fails. - GPU Serialiyes the execution
    for(int stride=1; stride < blockDim.x ; stride*=2){
        if(localId % (2*stride) == 0){
            sharedMem[localId] +=sharedMem[localId+stride];
        }
        __syncthreads();
    }

    if(localId==0){
        output[blockIdx.x] = sharedMem[0];
    }
}

__global__ void reduced_sum_sequential(float* input, float* output, int n){

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int localId = threadIdx.x;

    extern __shared__ float sharedMem[];

    if(id<n){
        sharedMem[localId] = input[id];
    }
    else{
        sharedMem[localId] = 0;
    }

    __syncthreads();

    // localId < stride ensures that all active threads are perfectly contiguous 
    //(e.g., threads 0 through 255 are active, while 256 through 512 are idle). 
    //Entire warps are either fully active or fully idle, meaning zero branch divergence within a warp.
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(localId < stride) {
            sharedMem[localId] += sharedMem[localId + stride];
        }
        __syncthreads();
    }

    if(localId==0){
        output[blockIdx.x] = sharedMem[0];
    }
}

__global__ void reduced_sum_sequential_shfl_down(float* input, float* output, int n) {
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x;

    extern __shared__ float sharedMemory[];

    if(id < n) {
        sharedMemory[localId] = input[id];
    } else {
        sharedMemory[localId] = 0;
    }
    __syncthreads();


    for(int stride = blockDim.x/2; stride >= 32; stride >>= 1) {
        if(localId < stride) {
            sharedMemory[localId] += sharedMemory[localId + stride];
        }
        __syncthreads();
    }

    if(localId < 32) {
        float sum = sharedMemory[localId];
        for(int offset = 16; offset > 0; offset >>= 1){
            //0xFFFFFFFF means all threads in the warp participate
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if(localId == 0) {
            output[blockIdx.x] = sum;
        }
    }

}

int main() {
   const int N         = 1024*1024;
    const int blockSize = 256;
    const int gridSize  = (N + blockSize - 1) / blockSize;
    const size_t size   = N * sizeof(float);

    float* h_input   = (float*)malloc(size);
    float* h_partial = (float*)malloc(gridSize * sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) cpu_sum += h_input[i];

    float *d_input, *d_output;
    cudaMalloc(&d_input,  size);
    cudaMalloc(&d_output, gridSize * sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    size_t sharedMem = blockSize * sizeof(float);

    auto verify = [&](const char* name) {
        cudaDeviceSynchronize();
        cudaMemcpy(h_partial, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
        float gpu_sum = 0.0f;
        for (int i = 0; i < gridSize; i++) {
            // std::cout<<h_partial[i]<<"\t";
            gpu_sum += h_partial[i];
        }
        printf("\n");
        printf("%s: %s\n", name, (fabsf(gpu_sum - cpu_sum) < 1.0f) ? "CORRECT" : "INCORRECT");
        printf("\n");
        cudaMemset(d_output, 0, gridSize * sizeof(float));
    };

    reduced_sum_naive<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_naive");

    reduced_sum_sequential<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_sequential");

    reduced_sum_sequential_shfl_down<<<gridSize, blockSize, sharedMem>>>(d_input, d_output, N);
    verify("reduced_sum_sequential_shfl_down");

    free(h_input);
    free(h_partial);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}