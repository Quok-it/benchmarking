#include <vector>
#include <iostream>
#include <cassert>
#include <cudnn.h>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>

// Constants
#define SIZE 256
#define MASK_WIDTH  3 // Note that mask refers to kernel
#define MASK_HEIGHT 3

// Function to pad the input matrix - to ensure that the output has the same size as input
void padInput(const std::vector<float>& input, std::vector<float>& paddedInput,
              int inputWidth, int inputHeight, int kernelWidth, int kernelHeight) {
    int padWidth = kernelWidth / 2;
    int padHeight = kernelHeight / 2;
    int paddedWidth = inputWidth + 2 * padWidth;
    int paddedHeight = inputHeight + 2 * padHeight;

    paddedInput.resize(paddedWidth * paddedHeight, 0.0f);

    for (int y = 0; y < inputHeight; ++y) {
        for (int x = 0; x < inputWidth; ++x) {
            paddedInput[(y + padHeight) * paddedWidth + (x + padWidth)] = input[y * inputWidth + x];
        }
    }
}

// Generate a matrix filled with random values
std::vector<float> generateRandomMatrix(int width, int height) {
    std::vector<float> matrix(width * height);
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &val : matrix) val = dist(rng);
    return matrix;
}

// Flip the kernel (for convolution)
void flipKernel(const std::vector<float>& kernel, std::vector<float>& flippedKernel,
                int kernelWidth, int kernelHeight) {
    for (int y = 0; y < kernelHeight; ++y) {
        for (int x = 0; x < kernelWidth; ++x) {
            flippedKernel[y * kernelWidth + x] = kernel[(kernelHeight - 1 - y) * kernelWidth + (kernelWidth - 1 - x)];
        }
    }
}

// Validate convolution with cuDNN
float benchmarkConvolution(const std::vector<float>& input, std::vector<float>& output_cudnn,
                         const std::vector<float>& kernel, int inputWidth, int inputHeight,
                         int kernelWidth, int kernelHeight) {
    int paddedWidth = inputWidth + 2 * (kernelWidth / 2);
    int paddedHeight = inputHeight + 2 * (kernelHeight / 2);
    float *input_device, *kernel_device, *output_device;
    const float alpha = 1.0f, beta = 0.0f;
    const int stride = 1;

    std::vector<float> flippedKernel(kernel.size());
    flipKernel(kernel, flippedKernel, kernelWidth, kernelHeight);
    
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, paddedHeight, paddedWidth);

    cudnnCreateFilterDescriptor(&kernelDesc);
    cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, kernelHeight, kernelWidth);

    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, inputHeight, inputWidth);

    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    int returnedAlgoCount;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, 1, &returnedAlgoCount, &algoPerf);
    cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;

    size_t workspaceSize = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize);
    void* workspace = nullptr;
    if (workspaceSize > 0) cudaMalloc(&workspace, workspaceSize);
    
    cudaMalloc(&input_device, input.size() * sizeof(float));
    cudaMalloc(&kernel_device, flippedKernel.size() * sizeof(float));
    cudaMalloc(&output_device, output_cudnn.size() * sizeof(float));
    cudaMemcpy(input_device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_device, flippedKernel.data(), flippedKernel.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudnnConvolutionForward(cudnn, &alpha, inputDesc, input_device, kernelDesc, kernel_device,
                            convDesc, algo, workspace, workspaceSize, &beta, outputDesc, output_device);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(output_cudnn.data(), output_device, inputWidth * inputHeight * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(workspace);
    cudaFree(input_device);
    cudaFree(kernel_device);
    cudaFree(output_device);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(kernelDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);   
    
    return milliseconds;
}

// Benchmark activation function
float benchmarkActivation(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inputDesc, float* input_device, float* output_device, int inputSize) {
    cudnnActivationDescriptor_t activationDesc;
    cudnnCreateActivationDescriptor(&activationDesc);
    cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudnnActivationForward(cudnn, activationDesc, &alpha, inputDesc, input_device, &beta, inputDesc, output_device);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudnnDestroyActivationDescriptor(activationDesc);
    return milliseconds;
}

// Benchmark pooling function
float benchmarkPooling(cudnnHandle_t cudnn, cudnnTensorDescriptor_t inputDesc, cudnnTensorDescriptor_t outputDesc, float* input_device, float* output_device) {
    cudnnPoolingDescriptor_t poolingDesc;
    cudnnCreatePoolingDescriptor(&poolingDesc);
    cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);

    const float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudnnPoolingForward(cudnn, poolingDesc, &alpha, inputDesc, input_device, &beta, outputDesc, output_device);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudnnDestroyPoolingDescriptor(poolingDesc);
    return milliseconds;
}

int main() {
    const int kernelWidth = MASK_WIDTH;
    const int kernelHeight = MASK_HEIGHT;

    const int inputWidth = SIZE;
    const int inputHeight = SIZE;
    std::vector<float> input = generateRandomMatrix(inputWidth, inputHeight);
    std::vector<float> paddedInput;
    padInput(input, paddedInput, inputWidth, inputHeight, kernelWidth, kernelHeight);
    std::vector<float> kernel(kernelWidth * kernelHeight, 1.0f);
    std::vector<float> output_cudnn(inputWidth * inputHeight, 0);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, inputHeight, inputWidth);
    
    float* input_device;
    float* output_device;
    cudaMalloc(&input_device, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&output_device, inputWidth * inputHeight * sizeof(float));
    cudaMemcpy(input_device, input.data(), inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);
    
    float cuDNN_time = benchmarkConvolution(paddedInput, output_cudnn, kernel, inputWidth, inputHeight, kernelWidth, kernelHeight);
    float activation_time = benchmarkActivation(cudnn, inputDesc, input_device, output_device, inputWidth * inputHeight);
    float pooling_time = benchmarkPooling(cudnn, inputDesc, inputDesc, input_device, output_device);

    std::cout << "Matrix Size: " << SIZE << "x" << SIZE << "\n";
    std::cout << "Conv Time: " << cuDNN_time << " ms\n";
    std::cout << "Activation Time: " << activation_time << " ms\n";
    std::cout << "Pooling Time: " << pooling_time << " ms\n";
    std::cout << "---------------------------------------------" << std::endl;
    
    cudaFree(input_device);
    cudaFree(output_device);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroy(cudnn);
    return 0;
}
