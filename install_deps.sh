#!/bin/bash

echo "Updating system packages..."
apt update && apt install -y \
    python3 \
    python3-pip \
    wget \
    cmake \
    g++ \
    curl \
    sudo \
    git

echo "Installing Python dependencies..."
pip3 install --no-cache-dir pandas openpyxl

echo "Installing CUDA & cuDNN dependencies..."
apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev

if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found! Installing CUDA Toolkit..."
    apt install -y nvidia-cuda-toolkit
fi

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo "CUDA PATH configured: $(which nvcc)"


echo "Installing AI Benchmark..."
pip3 install ai_benchmark
sed -i '/np.warnings.filterwarnings/d' /usr/local/lib/python3.12/site-packages/ai_benchmark/__init__.py

echo "Installing TensorFlow optimized for CUDA..."
pip3 install --no-cache-dir tensorflow[and-cuda]

if [ ! -f "cublas_benchmark.cu" ] || [ ! -f "cudnn_benchmark.cu" ]; then
    echo "ERROR: CUDA benchmark files not found! Ensure they are in the script directory."
    exit 1
fi
echo "Compiling CUBLAS benchmark..."
nvcc -o cublas_benchmark cublas_benchmark.cu -lcublas -lcudart

echo "Compiling CUDNN benchmark..."
nvcc cudnn_benchmark.cu -o cudnn_benchmark -lcudnn -lcuda -std=c++11

echo "Setting up entrypoint script..."
chmod +x entrypoint.sh

echo "Installation complete! To run the benchmark, execute:"
echo   entrypoint.sh