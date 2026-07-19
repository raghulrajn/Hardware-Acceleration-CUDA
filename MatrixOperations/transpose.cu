#include<iostream>


__global__ void transpose(float* input, float* output, size_t height, size_t width){

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(row< height && col< width){
        output[col * height + row] = input[row * width + col];
    }
}

#define tile_size 32
__global__ void transpose_sharedmem(float* input, float* output, size_t height, size_t width){

    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ float sharedMem[tile_size][tile_size];

    if(row< height && col< width){
        sharedMem[threadIdx.y][threadIdx.x]= input[row * width + col];
    }
    __syncthreads();

    int t_col = blockIdx.y*blockDim.y + threadIdx.x;
    int t_row = blockIdx.x*blockDim.x + threadIdx.y;

    if(t_row< width && t_col< height){
        output[t_row * height + t_col] = sharedMem[threadIdx.x][threadIdx.y];
    }
}


int main(){


    int N = 1024;
    float *h_input = (float*)malloc(N*N*sizeof(float));

    for(int i = 0; i<N*N; i++) h_input[i] = i;

    float* d_output, *d_input, *d_output2; 
    size_t size = N*N*sizeof(float);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_output2, size);
    float* h_output = (float*)malloc(size);
    float* h_output2 = (float*)malloc(size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    int WIDTH = N;
    int HEIGHT = N;
    dim3 blockDim(32,32);
    dim3 gridDim((WIDTH + blockDim.x - 1) / blockDim.x, (HEIGHT + blockDim.y - 1) / blockDim.y);

    transpose<<<gridDim, blockDim>>>(d_input, d_output, WIDTH, HEIGHT);
    transpose_sharedmem<<<gridDim, blockDim>>>(d_input, d_output2, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output2, d_output2, size, cudaMemcpyDeviceToHost);
    
    {bool correct = true;
    for(int row = 0; row < HEIGHT && correct; row++)
        for(int col = 0; col < WIDTH && correct; col++)
            if(h_input[row * WIDTH + col] != h_output[col * HEIGHT + row])
                correct = false;

     printf("%s\n",correct ? "CORRECT" : "INCORRECT");
    }
    bool correct = true;
    for(int row = 0; row < HEIGHT && correct; row++)
        for(int col = 0; col < WIDTH && correct; col++)
            if(h_input[row * WIDTH + col] != h_output2[col * HEIGHT + row])
                correct = false;
                
    printf("%s\n",correct ? "CORRECT" : "INCORRECT");
    free(h_input);
    free(h_output);
    free(h_output2);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output2);



}