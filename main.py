import os
import subprocess

def compile_cublas():
    print("\n>> Compiling CUBLAS Benchmark...")
    try:
        subprocess.run(["nvcc", "-o", "cublas_benchmark", "cublas_benchmark.cu", "-lcublas", "-lcudart"], check=True)
        print("CUBLAS Benchmark compiled successfully!")
    except subprocess.CalledProcessError as e:
        print(f"CUBLAS Compilation failed: {e}")
        exit(1)

def compile_cudnn():
    print("\n>> Compiling CUDNN Benchmark...")
    try:
        subprocess.run(["nvcc", "-o", "cudnn_benchmark", "cudnn_benchmark.cu", "-lcudnn", "-lcuda", "-std=c++11"], check=True)
        print("CUDNN Benchmark compiled successfully!")
    except subprocess.CalledProcessError as e:
        print(f"CUDNN Compilation failed: {e}")
        exit(1)

def run_cublas():
    print("\n>> Running CUBLAS Benchmark...")
    try:
        subprocess.run(["./cublas_benchmark"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"CUBLAS Benchmark failed: {e}")

def run_cudnn():
    print("\n>> Running CUDNN Benchmark...")
    try:
        subprocess.run(["./cudnn_benchmark"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"CUDNN Benchmark failed: {e}")

def run_ai_benchmark():
    print("\n>> Running AI Benchmark...")
    try:
        subprocess.run(["ai-benchmark"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"AI Benchmark failed: {e}")

if __name__ == "__main__":
    # Ensure GPU memory growth is enabled
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    # Compile benchmarks
    compile_cublas()
    compile_cudnn()

    # Run benchmarks
    run_cublas()
    run_cudnn()
    run_ai_benchmark()
