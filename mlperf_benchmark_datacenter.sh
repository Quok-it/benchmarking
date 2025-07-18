#!/bin/bash
set -euo pipefail


set -e 

# sudo apt-get update > /dev/null 2>&1
# sudo apt-get install -y python3.10-venv > /dev/null 2>&1
# sudo apt-get install -y python3-pip > /dev/null 2>&1
# sudo apt-get install -y --no-install-recommends \
#     libcudnn8 libcudnn8-dev > /dev/null 2>&1
# sudo apt install -y unzip wget curl > /dev/null 2>&1

# Activate a Virtual ENV for MLCFlow
python3 -m venv mlc
source mlc/bin/activate 

# Install packages
# pip install --upgrade pip
pip install mlc-scripts
pip install cmx4mlperf
# pip install --no-cache-dir pymongo
# pip install python-dotenv

# Install mlperf environment
mlcr install,python-venv --name=mlperf

# Set environment variable
export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"

# Run ResNet50 MLPerf inference test
# cr run-mlperf,inference,_find-performance,_full,_r5.0-dev \
#    --model=resnet50 \
#    --implementation=reference \
#    --framework=tensorflow \
#    --category=datacenter \
#    --scenario=Offline \
#    --execution_mode=test \
#    --device=cuda  \
#    --quiet \
#    --test_query_count=5000

# Run Stable Diffusion MLPerf inference test
# cr run-mlperf,inference,_find-performance,_full,_r5.0-dev \
#    --model=sdxl \
#    --implementation=reference \
#    --framework=pytorch \
#    --category=datacenter \
#    --scenario=Offline \
#    --execution_mode=test \
#    --device=cuda  \
#    --quiet \
#    --test_query_count=50

# Run BERT MLPerf inference test
cr run-mlperf,inference,_find-performance,_full,_r5.0-dev \
    --model=bert-99 \
    --implementation=reference \
    --framework=pytorch \
    --category=datacenter \
    --scenario=Offline \
    --execution_mode=test \
    --device=cuda  \
    --quiet \
    --test_query_count=500

# python3 parse.py

# Deactivate the virtual environment
deactivate