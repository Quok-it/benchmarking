import re
import json

def parse_benchmark_results(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    benchmark_data = {
        "CUBLAS matrix size": None,
        "CUBLAS execution time": None,
        "CUBLAS performance": None,
        "CuDNN matrix size": None,
        "Convolution time": None,
        "Activation time": None,
        "Pooling time": None,
        "MobileNet-V2": {},
        "Inception-V3": {},
        "Inception-V4": {},
        "Inception-ResNet-V2": {},
        "ResNet-V2-50": {},
        "ResNet-V2-152": {},
        "VGG-16": {},
        "SRCNN 9-5-5": {},
        "VGG-19 Super-Res": {},
        "ResNet-SRGAN": {},
        "ResNet-DPED": {},
        "U-Net": {},
        "Nvidia-SPADE": {},
        "ICNet": {},
        "PSPNet": {},
        "DeepLab": {},
        "Pixel-RNN": {}
    }

    # Extract CUBLAS info
    cublas_match = re.search(r"Matrix Size: (\d+x\d+)\nExecution Time: ([\d.]+) ms\nPerformance: ([\d.]+) GFLOPS", content)
    if cublas_match:
        benchmark_data["CUBLAS matrix size"] = cublas_match.group(1)
        benchmark_data["CUBLAS execution time"] = float(cublas_match.group(2))
        benchmark_data["CUBLAS performance"] = float(cublas_match.group(3))

    # Extract CuDNN info
    cudnn_match = re.search(r"Matrix Size: (\d+x\d+)\nConv Time: ([\d.]+) ms\nActivation Time: ([\d.]+) ms\nPooling Time: ([\d.]+) ms", content)
    if cudnn_match:
        benchmark_data["CuDNN matrix size"] = cudnn_match.group(1)
        benchmark_data["Convolution time"] = float(cudnn_match.group(2))
        benchmark_data["Activation time"] = float(cudnn_match.group(3))
        benchmark_data["Pooling time"] = float(cudnn_match.group(4))

    # Model performance parsing (previously correct approach)
    model_pattern = re.compile(r"(\d+)/\d+\.\s([^\n]+)")
    inference_pattern = re.compile(r"(\d+\.\d) - inference \| batch=(\d+), size=(\d+x\d+): ([\d.]+) ± [\d.]+ ms")
    training_pattern = re.compile(r"(\d+\.\d) - training \| batch=(\d+), size=(\d+x\d+): ([\d.]+) ± [\d.]+ ms")

    current_model = None

    for line in content.split("\n"):
        model_match = model_pattern.match(line)
        if model_match:
            current_model = model_match.group(2).strip()
        
        inference_match = inference_pattern.match(line)
        training_match = training_pattern.match(line)

        if inference_match and current_model in benchmark_data:
            benchmark_data[current_model]["inference"] = {
                "batch_size": int(inference_match.group(2)),
                "input_size": inference_match.group(3),
                "time_ms": float(inference_match.group(4))
            }

        if training_match and current_model in benchmark_data:
            benchmark_data[current_model]["training"] = {
                "batch_size": int(training_match.group(2)),
                "input_size": training_match.group(3),
                "time_ms": float(training_match.group(4))
            }

    return benchmark_data


# Example usage:
file_path = "results.txt"  # Update with your actual file path
benchmark_results = parse_benchmark_results(file_path)

# Print the extracted dictionary
print(json.dumps(benchmark_results, indent=4))
