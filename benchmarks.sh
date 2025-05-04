#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3.10-venv
sudo apt-get install -y python3-pip
sudo apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev
sudo apt install -y unzip wget curl

pip install --upgrade pip
pip install --no-cache-dir pymongo
pip install python-dotenv
python3 -m pip install --upgrade tensorrt
python3 -m pip install tensorrt-cu11 tensorrt-lean-cu11 tensorrt-dispatch-cu11

chmod +x mlperf_benchmark_datacenter.sh
chmod +x gpu_burn.sh
# chmod +x graphics_benchmark.sh
chmod +x nvidia_hpc_benchmark.sh
./mlperf_benchmark_datacenter.sh
./gpu_burn.sh
# ./graphics_benchmark.sh
./nvidia_hpc_benchmark.sh

python3 parse.py

