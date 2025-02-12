#!/bin/bash
set -e  # Stop script on first failure

# echo "Running CUBLAS Benchmark..."
# ./cublas_benchmark

# echo "Running CUDNN Benchmark..."
# ./cudnn_benchmark

# echo "Running AI Benchmark..."
chmod +x /usr/local/bin/ai-benchmark
# ai-benchmark

python3 main.py | tee results.txt
python3 parse.py | tee database_inputs.json

# Display output files to verify results
echo "==== RESULTS.TXT ===="
cat results.txt

echo "==== DATABASE_INPUTS.TXT ===="
cat database_inputs.json
