# Base image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set working directory
WORKDIR /benchmark

# Install dependencies
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    wget \
    cmake \
    g++ \
    curl \
    sudo \
    git

# Install basic dependencies
# RUN pip install numpy==1.23.5 
RUN pip install --no-cache-dir pandas openpyxl pymongo

# Install cuDNN
RUN apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev

# Install AI benchmark
RUN pip install ai_benchmark
RUN sed -i '/np.warnings.filterwarnings/d' /usr/local/lib/python3.10/dist-packages/ai_benchmark/__init__.py

# Install TensorFlow (Optimized for CUDA)
RUN pip install --no-cache-dir tensorflow[and-cuda]

# Copy AI benchmark script into the container
COPY ai_benchmark.py /benchmark/
COPY ai-benchmark-results.xlsx /benchmark/

# Copy CUBLAS and CUDNN benchmark scripts
COPY cublas_benchmark.cu cudnn_benchmark.cu /benchmark/

# Compile CUBLAS benchmark
RUN nvcc -o cublas_benchmark cublas_benchmark.cu -lcublas -lcudart

# Compile CUDNN benchmark
RUN nvcc cudnn_benchmark.cu -o cudnn_benchmark -lcudnn -lcuda -std=c++11

COPY main.py /benchmark/
COPY parse.py /benchmark/
COPY gpu_sanity_check.py /benchmark/
COPY ai-benchmark-results.py /benchmark/

# Set entrypoint script
COPY entrypoint.sh /benchmark/entrypoint.sh
RUN chmod +x /benchmark/entrypoint.sh

ENTRYPOINT ["/benchmark/entrypoint.sh"]
