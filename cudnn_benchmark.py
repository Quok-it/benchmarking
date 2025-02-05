import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import ctypes
import cudnn

# Initialize cuDNN
cudnn_handle = cudnn.cudnnCreate()

# Function to allocate GPU memory
def allocate_gpu_memory(shape, dtype=np.float32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    ptr = cuda.mem_alloc(size)
    return ptr

# Function to benchmark a cuDNN operation
def benchmark_cudnn_operation(operation_func, iterations=100):
    # Warm-up
    for _ in range(10):
        operation_func()
    
    start = time.time()
    for _ in range(iterations):
        operation_func()
    end = time.time()
    
    avg_time = (end - start) / iterations
    print(f"{operation_func.__name__}: {avg_time:.6f} sec per iteration")

# Convolution Benchmark
def conv_benchmark():
    in_channels, out_channels, height, width = 3, 64, 224, 224
    kernel_size = (3, 3)
    
    input_tensor = allocate_gpu_memory((1, in_channels, height, width))
    filter_tensor = allocate_gpu_memory((out_channels, in_channels, *kernel_size))
    output_tensor = allocate_gpu_memory((1, out_channels, height-2, width-2))
    
    conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
    cudnn.cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, cudnn.CUDNN_CROSS_CORRELATION)
    
    benchmark_cudnn_operation(lambda: cudnn.cudnnConvolutionForward(
        cudnn_handle, ctypes.c_float(1.0), input_tensor, filter_tensor,
        conv_desc, output_tensor, ctypes.c_float(0.0)
    ))

# Activation Benchmark
def relu_benchmark():
    shape = (1, 64, 224, 224)
    input_tensor = allocate_gpu_memory(shape)
    output_tensor = allocate_gpu_memory(shape)
    
    activation_desc = cudnn.cudnnCreateActivationDescriptor()
    cudnn.cudnnSetActivationDescriptor(activation_desc, cudnn.CUDNN_ACTIVATION_RELU, cudnn.CUDNN_PROPAGATE_NAN, 0)
    
    benchmark_cudnn_operation(lambda: cudnn.cudnnActivationForward(
        cudnn_handle, activation_desc, ctypes.c_float(1.0), input_tensor,
        ctypes.c_float(0.0), output_tensor
    ))

# Pooling Benchmark
def pooling_benchmark():
    shape = (1, 64, 224, 224)
    input_tensor = allocate_gpu_memory(shape)
    output_tensor = allocate_gpu_memory((1, 64, 112, 112))
    
    pooling_desc = cudnn.cudnnCreatePoolingDescriptor()
    cudnn.cudnnSetPooling2dDescriptor(pooling_desc, cudnn.CUDNN_POOLING_MAX, cudnn.CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2)
    
    benchmark_cudnn_operation(lambda: cudnn.cudnnPoolingForward(
        cudnn_handle, pooling_desc, ctypes.c_float(1.0), input_tensor,
        ctypes.c_float(0.0), output_tensor
    ))

if __name__ == "__main__":
    print("Benchmarking cuDNN Operations:")
    conv_benchmark()
    relu_benchmark()
    pooling_benchmark()
    
    # Destroy cuDNN handle
    cudnn.cudnnDestroy(cudnn_handle)
