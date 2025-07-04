#!/bin/bash
set -euo pipefail


# Might need more dependencies

git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
make
./gpu_burn | tee ~/benchmarking/gpu_burn.txt
cd ..
# Sample output (to parse)
# Run length not specified in the command line. Using compare file: compare.ptx
# Burning for 10 seconds.
# GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-f525f9aa-e74c-d44a-ca4c-4fd01af85add)
# Initialized device 0 with 81116 MB of memory (80523 MB available, using 72471 MB of it), using FLOATS
# Results are 268435456 bytes each, thus performing 281 iterations
# 70.0%  proc'd: 281 (46467 Gflop/s)   errors: 0   temps: 45 C
#         Summary at:   Mon Apr 28 01:27:45 EDT 2025

# 100.0%  proc'd: 281 (46467 Gflop/s)   errors: 0   temps: 45 C
#         Summary at:   Mon Apr 28 01:27:49 EDT 2025


# Killing processes with SIGTERM (soft kill)
# Freed memory for dev 0
# Uninitted cublas
# done

# Tested 1 GPUs:
#         GPU 0: OK
