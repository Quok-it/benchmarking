#!/bin/bash
set -e  # Stop script on first failure

# echo "Running CUBLAS Benchmark..."
# ./cublas_benchmark

# echo "Running CUDNN Benchmark..."
# ./cudnn_benchmark

# echo "Running AI Benchmark..."
chmod +x /usr/local/bin/ai-benchmark
# ai-benchmark

# nvidia-smi -L

python3 main.py | tee results.txt
python3 parse.py | tee gpu_benchmark_results.json
python3 ai-benchmark-results.py | tee ai_benchmark_results.json
python3 gpu_sanity_check.py 

# Display output files to verify results
echo "==== RESULTS.TXT ===="
cat results.txt

echo "==== GPU_BENCHMARK_RESULTS.JSON ===="
cat gpu_benchmark_results.json

echo "==== AI_BENCHMARK_RESULTS.JSON ===="
cat ai_benchmark_results.json
