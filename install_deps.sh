#!/bin/bash

echo "Updating system packages..."
apt update && apt install -y \
    python3-pip \
    wget \
    cmake \
    g++ \
    curl \
    sudo \
    git

echo "Installing Python dependencies..."
pip3 install --no-cache-dir pandas openpyxl

echo "Installing AI Benchmark..."
pip3 install ai_benchmark
sed -i '/np.warnings.filterwarnings/d' /usr/local/lib/python3.12/site-packages/ai_benchmark/__init__.py

echo "Installing TensorFlow optimized for CUDA..."
pip3 install --no-cache-dir tensorflow[and-cuda]
