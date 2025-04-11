import re
import json
import os
import psycopg2
from datetime import datetime, timezone
import subprocess

# Connect to Postgres database
conn = psycopg2.connect(
    dbname="senior_design",
    user="postgres",
    password="quokit2025!",
    host="backend-postresql.cmfkgwq6gehx.us-east-1.rds.amazonaws.com"
)

cursor = conn.cursor()

def parse_benchmark_results(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    benchmark_data = {
        # "CUBLAS" : {
        #     "Matrix size": None,
        #     "Execution time": None,
        #     "Performance": None,
        # },
        # "cuDNN" : {
        #     "Matrix size": None,
        #     "Convolution time": None,
        #     "Activation time": None,
        #     "Pooling time": None,   
        # },
        "DL" : {
            "MobileNet-V2": {},
            "Inception-V3": {},
            "Inception-V4": {},
            "Inception-ResNet-V2": {},
            "ResNet-V2-50": {},
            "ResNet-V2-152": {},
            "VGG-16": {},
            "Nvidia-SPADE": {},
            "ICNet": {},
            "PSPNet": {},
            "DeepLab": {},
            "Pixel-RNN": {},
        }
    }

    # Extract CUBLAS results
    # cublas_match = re.search(r"Matrix Size: (\d+x\d+)\nExecution Time: ([\d.]+) ms\nPerformance: ([\d.]+) GFLOPS", content)
    # if cublas_match:
    #     benchmark_data["CUBLAS"]["Matrix size"] = cublas_match.group(1)
    #     benchmark_data["CUBLAS"]["Execution time"] = float(cublas_match.group(2))
    #     benchmark_data["CUBLAS"]["Performance"] = float(cublas_match.group(3))

    # # Extract CuDNN results
    # cudnn_match = re.search(r"Matrix Size: (\d+x\d+)\nConv Time: ([\d.]+) ms\nActivation Time: ([\d.]+) ms\nPooling Time: ([\d.]+) ms", content)
    # if cudnn_match:
    #     benchmark_data["cuDNN"]["Matrix size"] = cudnn_match.group(1)
    #     benchmark_data["cuDNN"]["Convolution time"] = float(cudnn_match.group(2))
    #     benchmark_data["cuDNN"]["Activation time"] = float(cudnn_match.group(3))
    #     benchmark_data["cuDNN"]["Pooling time"] = float(cudnn_match.group(4))

    # Extract Deep Learning results 
    model_pattern = re.compile(r"(\d+)/\d+\.\s([^\n]+)")
    inference_pattern = re.compile(r"\d+\.\d - inference \| batch=(\d+), size=(\d+x\d+): ([\d.]+).*ms")
    training_pattern = re.compile(r"\d+\.\d - training\s+\| batch=(\d+), size=(\d+x\d+): ([\d.]+).*ms")
    current_model = None

    for line in content.split("\n"):
        model_match = model_pattern.match(line)
        if model_match:
            current_model = model_match.group(2).strip()
        
        inference_match = inference_pattern.match(line)
        training_match = training_pattern.match(line)

        if inference_match and current_model in benchmark_data["DL"]:
            benchmark_data["DL"][current_model]["Inference time"] = float(inference_match.group(3))
        if training_match and current_model in benchmark_data["DL"]:
            benchmark_data["DL"][current_model]["Training time"] = float(training_match.group(3))
    
    return benchmark_data

def store_benchmark_results(file_path):
    """ Parses the benchmark results and stores them in MongoDB. """
    benchmark_results = parse_benchmark_results(file_path)
    print(json.dumps(benchmark_results, indent=4))

    # Add metadata (timestamp & Unix time)
    timestamp = datetime.now(timezone.utc)
    unix_time = timestamp.timestamp()

    def get_gpu_uuids():
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True
            ).strip()
            uuids = output.splitlines()
            return uuids
        except subprocess.CalledProcessError as e:
            print("Failed to query GPU UUIDs:", e)
            return []

    gpu_uuid = get_gpu_uuids()[0] if get_gpu_uuids() else "unknown"
    try:
        cursor.execute("""
            INSERT INTO benchmarks (gpu_uuid, timestamp, benchmark_data)
            VALUES (%s, %s, %s)
            ON CONFLICT (gpu_uuid, timestamp) DO NOTHING;
        """, (
            gpu_uuid,
            timestamp,
            json.dumps(benchmark_results)
        ))
        conn.commit()
        print("Benchmark results inserted.")
    except Exception as e:
        print("Error inserting into PostgreSQL:", e)
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
    

# Example usage:
file_path = "results.txt"  # Update with your actual file path
store_benchmark_results(file_path)
