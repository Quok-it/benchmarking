#include <iostream>
#include <random>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

std::default_random_engine generator(2);
std::uniform_real_distribution<float> distribution(0, 1);

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))

#define SIZE 32

void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution(generator);
    }
}

void const_init_matrix(float *mat, int N, float F)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = F;
    }
}

void cublasErrorCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line,
               cublasGetStatusName(status), cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
    float *A, float *B, float beta, float *C)
{
    cublasStatus_t ok = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cublasCheck(ok);
}

int main() {
    // Setup cublas
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    uint16_t m = SIZE, n = SIZE, k = SIZE;

    // GEMM computes C = α*AB+β*C

    // Just do pure A*B+C (for simpler debugging)
    float alpha = 1.0, beta = 1.0, initC = 1.0;
    float *A = nullptr, *B=nullptr, *C = nullptr, *C_ref = nullptr;     // host matrices
    float *dA = nullptr, *dB=nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * SIZE * SIZE);
    B = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C_ref = (float *)malloc(sizeof(float) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    // randomize_matrix(C, SIZE * SIZE);

    const_init_matrix(C, SIZE * SIZE, initC);

    // A, B, C live in CPU, dA, dB, dC live in GPU
    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * SIZE * SIZE));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));

    // Start timing
    cudaCheck(cudaEventRecord(start));

    runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

    // Stop timing
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));

    // Compute GFLOPS (2 * SIZE^3 FLOPs per matrix multiplication)
    double gflops = (2.0 * SIZE * SIZE * SIZE) / (milliseconds * 1e6);
    // Giga Floating Point Operations Per Second
    
    std::cout << "Matrix Size: " << SIZE << "x" << SIZE << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    // Free CPU and GPU memory
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));
    cudaCheck(cudaFree(dC_ref));
    cublasCheck(cublasDestroy(handle));

    return 0;

}