#!/bin/bash
set -e

# TensorRT .deb from Google Drive
FILE_ID="1F-cfPa_NUjI0bq2cRQce5oxep3DdPlnm"
DEB_NAME="nv-tensorrt-local-repo.deb"
TAR_PATH="/tmp/tensorrt.tar.gz"

# Download .deb using wget (fallback to gdown if available)
if ! command -v gdown &>/dev/null; then
  echo "📥 Downloading TensorRT .deb from Google Drive via wget..."
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt \
    --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=${FILE_ID}" -O- \
    | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')

  wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" \
    -O "$DEB_NAME"
  rm -f /tmp/cookies.txt
else
  echo "📥 Downloading TensorRT .deb from Google Drive via gdown..."
  gdown --id "$FILE_ID" -O "$DEB_NAME"
fi

# Install TensorRT local APT repo and SDK
sudo dpkg -i "$DEB_NAME"
sudo cp /var/nv-tensorrt-local-repo*/**.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y tensorrt

# Package headers/libs for MLPerf
mkdir -p /tmp/tensorrt/include /tmp/tensorrt/lib
cp -r /usr/include/Nv*.h /tmp/tensorrt/include/
cp /usr/lib/x86_64-linux-gnu/libnvinfer* /tmp/tensorrt/lib/
cp /usr/lib/x86_64-linux-gnu/libnvonnxparser* /tmp/tensorrt/lib/
tar -czf "$TAR_PATH" -C /tmp tensorrt
mlcr get,tensorrt,_dev --tar_file="$TAR_PATH"

# Set up MLPerf virtual env
python3 -m venv mlc
source mlc/bin/activate
pip install "numpy<2" --force-reinstall
pip install mlc-scripts cmx4mlperf

mlcr install,python-venv --name=mlperf
export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"

# Run MLPerf Inference Benchmarks
cr run-mlperf,inference,_find-performance,_full,_r5.0-dev \
   --model=resnet50 \
   --implementation=nvidia \
   --framework=tensorrt \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=cuda \
   --quiet \
   --test_query_count=500

cr run-mlperf,inference,_find-performance,_full,_r5.0-dev \
   --model=bert-99 \
   --implementation=nvidia \
   --framework=tensorrt \
   --category=datacenter \
   --scenario=Offline \
   --execution_mode=test \
   --device=cuda \
   --quiet \
   --test_query_count=500

deactivate
