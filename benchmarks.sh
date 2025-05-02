#!/bin/bash

sudo apt-get update > /dev/null 2>&1
sudo apt-get install -y python3.10-venv > /dev/null 2>&1
sudo apt-get install -y python3-pip > /dev/null 2>&1
sudo apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev > /dev/null 2>&1
sudo apt install -y unzip wget curl > /dev/null 2>&1

./mlperf_benchmark_datacenter.sh
./gpu_burn.sh
# ./graphics_benchmark.sh
./nvidia_hpc_benchmark.sh

