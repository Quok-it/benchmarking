import re
import json

def parse_benchmark_results(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    benchmark_data = {
        "CUBLAS" : {
            "Matrix size": None,
            "Execution time": None,
            "Performance": None,
        },
        "cuDNN" : {
            "Matrix size": None,
            "Convolution time": None,
            "Activation time": None,
            "Pooling time": None,   
        },
        "DL" : {
            "MobileNet-V2": {
                "Inference time": None,
                "Training time": None
                },
            "Inception-V3": {
                "Inference time": None,
                "Training time": None
                },
            "Inception-V4": {
                "Inference time": None,
                "Training time": None
                },
            "Inception-ResNet-V2": {
                "Inference time": None,
                "Training time": None
                },
            "ResNet-V2-50": {
                "Inference time": None,
                "Training time": None
                },
            "ResNet-V2-152": {
                "Inference time": None,
                "Training time": None
                },
            "VGG-16": {
                "Inference time": None,
                "Training time": None
                },
            "Nvidia-SPADE": {
                "Inference time": None,
                "Training time": None
                },
            "ICNet": {
                "Inference time": None,
                "Training time": None
                },
            "PSPNet": {
                "Inference time": None,
                "Training time": None
                },
            "DeepLab": {
                "Inference time": None,
                "Training time": None
                },
            "Pixel-RNN": {
                "Inference time": None,
                "Training time": None
                },
        }
    }

    # Extract CUBLAS results
    cublas_match = re.search(r"Matrix Size: (\d+x\d+)\nExecution Time: ([\d.]+) ms\nPerformance: ([\d.]+) GFLOPS", content)
    if cublas_match:
        benchmark_data["CUBLAS"]["Matrix size"] = cublas_match.group(1)
        benchmark_data["CUBLAS"]["Execution time"] = float(cublas_match.group(2))
        benchmark_data["CUBLAS"]["Performance"] = float(cublas_match.group(3))

    # Extract CuDNN results
    cudnn_match = re.search(r"Matrix Size: (\d+x\d+)\nConv Time: ([\d.]+) ms\nActivation Time: ([\d.]+) ms\nPooling Time: ([\d.]+) ms", content)
    if cudnn_match:
        benchmark_data["cuDNN"]["Matrix size"] = cudnn_match.group(1)
        benchmark_data["cuDNN"]["Convolution time"] = float(cudnn_match.group(2))
        benchmark_data["cuDNN"]["Activation time"] = float(cudnn_match.group(3))
        benchmark_data["cuDNN"]["Pooling time"] = float(cudnn_match.group(4))

    # Extract Deep Learning results 
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

        if inference_match and current_model in benchmark_data["DL"]:
            benchmark_data["DL"][current_model]["Inference time"] = float(inference_match.group(4))

        if training_match and current_model in benchmark_data["DL"]:
            benchmark_data["DL"][current_model]["Training time"] = float(training_match.group(4))

    return benchmark_data

# Example usage:
file_path = "results.txt"  # Update with your actual file path
benchmark_results = parse_benchmark_results(file_path)

# Print the extracted dictionary
print(json.dumps(benchmark_results, indent=4))
