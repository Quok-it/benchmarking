## Strategy
Do low-level benchmarking in C, to measure raw GPU performance and do high-level benchmarking in Python to test real-world training and inference. Use python to automate C++ compile & run.

## MLPerf
wsl --install

In WSL,
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html 

Pull docker image,
docker pull nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v4.1-cuda12.4-pytorch24.04-ubuntu22.04-x86_64-release

Run the container giving MLPerf access to gpus
docker run --gpus all -it --rm nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v4.1-cuda12.4-pytorch24.04-ubuntu22.04-x86_64-release 

docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    nvcr.io/nvidia/mlperf/mlperf-inference:mlpinf-v4.1-cuda12.4-pytorch24.04-ubuntu22.04-x86_64-release

apt update
apt install -y python3-venv

Set up MLCommons Automation Framework,
python3 -m venv mlperf-venv
source mlperf-venv/bin/activate
pip install mlc-scripts

Run MLPerf benchmark,
mlcr run-mlperf,inference,_r5.0-dev \
   --model=resnet50 \
   --implementation=reference \
   --framework=onnxruntime \
   --category=edge \
   --scenario=Offline \
   --execution_mode=valid \
   --device=cuda \
   --quiet
   
   https://docs.mlcommons.org/inference/benchmarks/image_classification/resnet50/#__tabbed_8_1

## AI Benchmarks
https://pypi.org/project/ai-benchmark/ 
ai-benchmark 
or
python ai_benchmark.py

## CUBLAS Benchmark
nvcc -o cublas_benchmark cublas_benchmark.cu -lcublas -lcudart
./cublas_benchmark.exe

## CUDNN Benchmarks
nvcc cudnn_benchmark.cu -o cudnn_benchmark -lcudnn -lcuda -std=c++11
./cudnn_benchmark.exe

## Notes
Not sure why but I'm unable to run MLPerf, this is the error: ! call C:\Users\tarun\CM\repos\mlcommons@mlperf-automations\script\get-llvm\run.bat from tmp-run.bat
The system cannot find the path specified.

Trying using docker environment instead:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html 
