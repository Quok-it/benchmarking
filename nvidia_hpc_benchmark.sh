#!/bin/bash

cd .. 
wget https://developer.download.nvidia.com/compute/nvidia-hpc-benchmarks/25.04/local_installers/nvidia-hpc-benchmarks-local-repo-ubuntu2204-25.04_1.0-1_amd64.deb
sudo dpkg -i nvidia-hpc-benchmarks-local-repo-ubuntu2204-25.04_1.0-1_amd64.deb
sudo cp /var/nvidia-hpc-benchmarks-local-repo-ubuntu2204-25.04/nvidia-hpc-benchmarks-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt install nvidia-hpc-benchmarks-openmpi
sudo apt install libopenmpi-dev
cd /opt/nvidia/nvidia_hpc_benchmarks_openmpi
./hpl.sh --dat ./hpl-linux-x86_64/sample-dat/HPL-1GPU.dat | tee ~/benchmarking/hpl_results.txt
./hpcg.sh --dat ./hpl-linux-x86_64/sample-dat/HPL-1GPU.dat --nx 256 --ny 256 --nz 256 --rt 2 | tee ~/benchmarking/hpcg_results.txt
./stream-gpu-test.sh --d 0 --n 100000000 | tee ~/benchmarking/stream_results.txt
# cd ..
# cd benchmarking