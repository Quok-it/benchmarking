#!/bin/bash
echo "Running CUBLAS Benchmark..."
./cublas_benchmark

echo "Running CUDNN Benchmark..."
./cudnn_benchmark

echo "Running AI Benchmark..."
chmod +x /usr/local/bin/ai-benchmark
ai-benchmark

