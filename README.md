# This repository provides automated scripts to benchmark:
## MLPerf Inference workloads (e.g., ResNet50, BERT)
**MLPerf Inference** is an industry-standard benchmark suite developed by MLCommons to measure the performance of machine learning inference across different hardware and software platforms.
* **ResNet50**: A convolutional neural network model used for image classification tasks. It assesses the GPU's ability to process and classify images efficiently.
* **BERT-99**: A transformer-based model for natural language processing tasks, such as question answering. It evaluates the GPU's performance in handling complex language models.
## GPU stress testing via gpu_burn
**GPU Burn** is a stress-testing tool designed to push the GPU to its limits by performing intensive computations. It helps in identifying stability issues and thermal performance under maximum load.
## Graphics performance using phoronix (not supported right now)
## High-performance computing (HPC) workloads
  https://docs.nvidia.com/nvidia-hpc-benchmarks/
NVIDIA provides a set of High-Performance Computing (HPC) benchmarks to evaluate the computational capabilities of GPUs in scientific and engineering applications. 
* **HPL (High-Performance Linpack)**: Measures a system's floating-point computing power by solving a dense system of linear equations. It's commonly used to rank supercomputers in the TOP500 list.
* **HPCG (High-Performance Conjugate Gradient)**: Assesses a system's performance in solving sparse linear systems, which are more representative of real-world applications than HPL.
* **STREAM**: Evaluates memory bandwidth by performing simple vector operations. It's essential for understanding how quickly data can be moved to and from the GPU's memory.
These benchmarks provide insights into a GPU's suitability for various HPC workloads, from numerical simulations to data-intensive tasks.
