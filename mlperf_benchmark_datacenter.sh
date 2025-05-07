#!/bin/bash
set -e

# Ensure ~/.local/bin is in PATH for gdown
export PATH="$HOME/.local/bin:$PATH"

# === Download TensorRT .deb from Google Drive ===
FILE_ID="1F-cfPa_NUjI0bq2cRQce5oxep3DdPlnm"
DEB_NAME="nv-tensorrt-local-repo.deb"

# Install gdown if not available
if ! command -v gdown &>/dev/null; then
  echo "⚠️ Installing gdown..."
  pip install --user gdown
fi

echo "📥 Downloading TensorRT .deb using gdown..."
gdown --id "$FILE_ID" -O "$DEB_NAME"

# === Install TensorRT local repo ===
echo "📦 Installing TensorRT local APT repo..."
sudo dpkg -i "$DEB_NAME"
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.10.0-cuda-11.8/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y tensorrt libnvinfer-dev libnvonnxparsers-dev

# === Bundle headers and libs ===
PACKAGE_DIR="$HOME/tensorrt_package"
mkdir -p "$PACKAGE_DIR/include" "$PACKAGE_DIR/lib"

# Copy headers and libraries (from multiarch path)
cp /usr/include/x86_64-linux-gnu/Nv*.h "$PACKAGE_DIR/include/"
cp /usr/lib/x86_64-linux-gnu/libnvinfer*.so* "$PACKAGE_DIR/lib/"
cp /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so* "$PACKAGE_DIR/lib/"

# Tar it up for MLPerf
TAR_PATH="$PACKAGE_DIR.tar.gz"
tar -czf "$TAR_PATH" -C "$HOME" tensorrt_package
mlcr get,tensorrt,_dev --tar_file="$TAR_PATH"

# === Setup Python env ===
echo "🐍 Creating Python venv..."
python3 -m venv mlc
source mlc/bin/activate
pip install --upgrade pip
pip install "numpy<2" mlc-scripts cmx4mlperf

mlcr install,python-venv --name=mlperf
export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf"

# === Run benchmarks ===
echo "🚀 Running MLPerf benchmarks..."

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
echo "✅ Done. Artifacts saved to: $PACKAGE_DIR and $TAR_PATH"
