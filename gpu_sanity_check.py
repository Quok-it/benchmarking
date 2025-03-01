import json
import math
import re
import subprocess
import pymongo
from pymongo import MongoClient
from datetime import datetime, timezone

# connect to MongoDB Atlas
mongo_uri = "mongodb+srv://shemilyshen:3g6wfTcdh7HS9ZGF@quokmvp.y18rg.mongodb.net/?retryWrites=true&w=majority&appName=QuokMVP"
if not mongo_uri:
    raise ValueError(f"MongoDB URI is not set in environmen variables.")

try:
    client = MongoClient(mongo_uri)
    db = client["gpu_monitoring"]
    client.admin.command('ping')  # Ensure connection is live
except pymongo.errors.PyMongoError as e:
    exit(1)

def gpu_sanity_check(gpu_model):
    # Load AI benchmark results
    with open('ai_benchmark_results.json', 'r') as file:
        ai_benchmark_results = json.load(file)

    # Load GPU benchmark results
    with open('gpu_benchmark_results.json', 'r') as file:
        gpu_benchmark_results = json.load(file)

    failed_count = 0  # Counter to track failed tests
    sanity_results = {}

    for network_name in gpu_benchmark_results["DL"].keys():
        if network_name not in ai_benchmark_results:
            print(f"Skipping '{network_name}': Not found in AI benchmark results.")
            continue
        
        # Ensure GPU model exists in AI benchmark results
        if gpu_model not in ai_benchmark_results[network_name]["Inference time"]:
            print(f"GPU model '{gpu_model}' not found for '{network_name}' in AI benchmark results.")
            failed_count += 1  # Count failure
            sanity_results[network_name] = "missing data"
            continue  # Skip this iteration

        # Extract values safely
        if len(gpu_benchmark_results["DL"][network_name].keys()) != 0:
            gpu_inference = gpu_benchmark_results["DL"][network_name]["Inference time"]
            gpu_training = gpu_benchmark_results["DL"][network_name]["Training time"]

        ai_inference = ai_benchmark_results[network_name]["Inference time"].get(gpu_model, None)
        ai_training = ai_benchmark_results[network_name]["Training time"].get(gpu_model, None)

        # Ensure AI benchmark values exist
        if ai_inference is None or ai_training is None:
            print(f"Missing benchmark data for '{gpu_model}' in '{network_name}'")
            failed_count += 1  # Count failure
            sanity_results[network_name] = "missing data"
            continue

        # Check if values are within tolerance
        if ai_inference is not None and ai_training is not None and gpu_inference is not None and gpu_training is not None:
            if not (math.isclose(gpu_inference, ai_inference, abs_tol=50) and
                    math.isclose(gpu_training, ai_training, abs_tol=50)):
                print(f"Your GPU '{gpu_model}' failed the sanity benchmarking test for '{network_name}'.")
                failed_count += 1  # Count failure
                sanity_results[network_name] = "failure"
            else:
                print(f"Your GPU '{gpu_model}' passed the sanity benchmarking test for '{network_name}'.")
                sanity_results[network_name] = "passed"
    
    # Final Results
    if failed_count > 0:
        print(f"\nYour GPU failed {failed_count} test(s) in the sanity benchmarking test!")
    else:
        print("\nYour GPU passed all sanity benchmarking tests!")
    
    timestamp = datetime.now(timezone.utc)
    unix_time = timestamp.timestamp()

    # Insert into MongoDB collection
    result = db.gpu_sanity_results.update_one(
        {"gpu model": gpu_model, "timestamp": timestamp.isoformat(), "unix_time": unix_time},
        {"$set": {
            "sanity_results": sanity_results
        }}, upsert=True)

output = subprocess.check_output(["nvidia-smi", "-L"], text=True).strip()

# Regular expression to match all GPU names
gpu_matches = re.findall(r"NVIDIA\s+(GeForce\s+RTX\s+\d+)", output)

if gpu_matches:
    gpu_name = gpu_matches[0]
    # for gpu_name in gpu_matches:
    print(f"Checking {gpu_name}")
    gpu_sanity_check(gpu_name)
else:
    print("No NVIDIA GPUs detected.")

