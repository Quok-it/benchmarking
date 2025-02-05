## Command to compile & run cublas_benchmark.cu:
"nvcc -o cublas_benchmark cublas_benchmark.cu -lcublas -lcudart"
"./cublas_benchmark.exe"

## Strategy:
Do low-level benchmarking in C, to measure raw GPU performance and do high-level benchmarking in Python to test real-world training and inference. Use python to automate C++ compile & run.