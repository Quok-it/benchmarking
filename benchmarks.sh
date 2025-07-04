#!/bin/bash
set -euo pipefail


sudo apt-get update
sudo apt-get install -y python3.10-venv
sudo apt-get install -y python3-pip
sudo apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev
sudo apt-get install -y unzip wget curl

# GPU Burn Dependencies (pulls gcc, stdc headers, glibc headers, dpkg stuff, and make)
sudo apt-get install -y build-essential

# Python module dependencies, never use pip without a venv
sudo apt-get install -y python3-pymongo python3-dotenv

chmod +x mlperf_benchmark_datacenter.sh
chmod +x gpu_burn.sh
# chmod +x graphics_benchmark.sh
chmod +x nvidia_hpc_benchmark.sh
./mlperf_benchmark_datacenter.sh
./gpu_burn.sh
# ./graphics_benchmark.sh
./nvidia_hpc_benchmark.sh

python3 parse.py

