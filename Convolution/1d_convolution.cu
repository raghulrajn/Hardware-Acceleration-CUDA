#include<iostream>
#include<vector> 
#include<math.h>
#include <cuda_runtime.h>

#define MAX_KERNEL_SIZE 5

__constant__ float d_kernel_const[MAX_KERNEL_SIZE];

__global__
void conv1d_constant_kernel(const float* input,
                   float* output,
                   int n,
                   int k,
                   int padding,
                   int output_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output_size) return;

    float sum = 0.0f;

    for (int j = 0; j < k; j++) {
        int input_index = i - padding + j;

        if (input_index >= 0 && input_index < n) {
            sum += input[input_index] * d_kernel_const[j];
        }
    }

    output[i] = sum;
}

__global__
void conv1d_kernel(const float* input,
                   const float* kernel,
                   float* output,
                   int n,         
                   int k,   
                   int padding,
                   int output_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output_size) return;

    float sum = 0.0f;

    for (int j = 0; j < k; j++) {
        int input_index = i - padding + j;

        if (input_index >= 0 && input_index < n) {
            sum += input[input_index] * kernel[j];
        }
    }

    output[i] = sum;
}

std::vector<float> conv_1d(const std::vector<float>& input,
                           const std::vector<float>& kernel,
                           int padding = 0,
                           int stride = 1,
                           int dilation = 1)
{
    int n = input.size();
    int k = kernel.size();

    int output_size =
        ((n + 2 * padding - dilation * (k - 1) - 1) / stride) + 1;

    std::vector<float> out(output_size, 0.0f);

    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;

        for (int j = 0; j < k; j++) {
            int input_index = i * stride - padding + j * dilation;

            if (input_index >= 0 && input_index < n) {
                sum += input[input_index] * kernel[j];
            }
        }

        out[i] = sum;
    }

    return out;
}

int main(){
    std::vector<float> input = {1,2,3,4,5,6,7};
    std::vector<float> kernel = {8,1,5,7,3};
    std::vector<float> out = conv_1d(input, kernel);

    for(auto i:out)
    std::cout<< i<< " ";

    int n = 7;
    int k = 5;
    int padding = 0;

    float h_input[7]  = {1,2,3,4,5,6,7};
    float h_kernel[5] = {8,1,5,7,3};

    int output_size = (n + 2*padding - k) + 1;

    float *d_input, *d_kernel, *d_output, *d_output1;

    cudaMalloc(&d_input,  n * sizeof(float));
    cudaMalloc(&d_kernel, k * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_output1, output_size * sizeof(float));

    cudaMemcpy(d_input,  h_input,  n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, k * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_kernel_const, h_kernel, k * sizeof(float));

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;


    conv1d_kernel<<<blocks, threads>>>(d_input, d_kernel, d_output, n, k, padding, output_size);

    float h_output[3];
    cudaMemcpy(h_output, d_output,
               output_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    std::cout<<"\n";
    for (int i = 0; i < output_size; i++)
        std::cout << h_output[i] << " ";
    
    std::cout<<"\n";
    conv1d_constant_kernel<<<blocks, threads>>>(
                    d_input,
                    d_output1,
                    n,
                    k,
                    padding,
                    output_size
                );

    cudaMemcpy(h_output, d_output1,output_size * sizeof(float),cudaMemcpyDeviceToHost);

    for (int i = 0; i < output_size; i++)
        std::cout << h_output[i] << " ";
    std::cout<<"\n";
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}