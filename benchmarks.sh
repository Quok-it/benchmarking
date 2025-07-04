#!/bin/bash

sudo apt-get update
sudo apt-get install -y python3.10-venv
sudo apt-get install -y python3-pip
sudo apt-get install -y --no-install-recommends \
    libcudnn8 libcudnn8-dev
sudo apt install -y unzip wget curl

pip install --upgrade pip
pip install -r requirements.txt

chmod +x mlperf_benchmark_datacenter.sh
chmod +x gpu_burn.sh
# chmod +x graphics_benchmark.sh
chmod +x nvidia_hpc_benchmark.sh

# Initialize database if .env file exists
if [ -f ".env" ]; then
    echo "Initializing database..."
    python3 init_database.py
else
    echo "Warning: .env file not found. Please run setup_rds.sh first or create .env manually."
fi

./mlperf_benchmark_datacenter.sh
./gpu_burn.sh
# ./graphics_benchmark.sh
./nvidia_hpc_benchmark.sh

python3 parse.py

