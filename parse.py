import os
import pymongo
from pymongo import MongoClient
from datetime import datetime, timezone
import glob
from dotenv import load_dotenv

load_dotenv()

# Setup MongoDB connection
try:
    client = MongoClient(os.environ["MONGODB_URI"])
    db = client["gpu_monitoring"]
    client.admin.command('ping')  # Ensure connection is live
except pymongo.errors.PyMongoError as e:
    print(f"MongoDB connection error: {e}")
    exit(1)

def parse_mlperf_result(file_path, model_name):
    samples_per_sec = None
    mean_latency_ns = None

    with open(file_path, 'r') as f:
        for line in f:
            if "Samples per second" in line:
                samples_per_sec = float(line.strip().split(":")[1])
            if "Mean latency (ns)" in line:
                mean_latency_ns = float(line.strip().split(":")[1])

    if samples_per_sec is None or mean_latency_ns is None:
        raise ValueError(f"Could not find required fields in {file_path}")

    mean_latency_sec = mean_latency_ns / 1e9

    print(f"Model: {model_name}")
    print(f"  - Samples per second: {samples_per_sec:.3f} samples/sec")
    print(f"  - Mean latency: {mean_latency_ns:.0f} ns ({mean_latency_sec:.3f} seconds)")

    return samples_per_sec, mean_latency_sec

def find_latest_run(model_dirs):
    """Find the latest mlperf_log_summary.txt across multiple test_results"""
    latest_summary = None
    latest_mtime = -1

    for model_dir in model_dirs:
        offline_perf_dir = os.path.join(model_dir, "offline/performance")
        run_dirs = glob.glob(os.path.join(offline_perf_dir, "run_*"))
        for run_dir in run_dirs:
            summary_path = os.path.join(run_dir, "mlperf_log_summary.txt")
            if os.path.exists(summary_path):
                mtime = os.path.getmtime(summary_path)
                if mtime > latest_mtime:
                    latest_summary = summary_path
                    latest_mtime = mtime

    if latest_summary is None:
        raise ValueError("No mlperf_log_summary.txt found for model.")
    
    return latest_summary

def find_all_test_result_paths():
    cache_dir = os.path.expanduser("~/MLC/repos/local/cache/")
    candidates = glob.glob(os.path.join(cache_dir, "get-mlperf-inference-results-*"))
    if not candidates:
        raise ValueError(f"No get-mlperf-inference-results-* directories found in {cache_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple get-mlperf-inference-results-* directories found, please clean up: {candidates}")

    results_root = candidates[0]
    test_results_dirs = glob.glob(os.path.join(results_root, "test_results/*"))
    if not test_results_dirs:
        raise ValueError(f"No test_results found under {results_root}")

    print(f"Auto-detected {len(test_results_dirs)} test_results folders under {results_root}")
    return test_results_dirs

if __name__ == "__main__":
    test_results_dirs = find_all_test_result_paths()

    models_list = ["resnet50", "stable-diffusion-xl", "bert-99"]

    benchmark_results = {}

    for model_name in models_list:
        try:
            model_dirs = [os.path.join(test_result_dir, model_name) for test_result_dir in test_results_dirs]
            summary_file = find_latest_run(model_dirs)
            samples_per_sec, mean_latency_sec = parse_mlperf_result(summary_file, model_name)

            benchmark_results[model_name] = {
                "Samples per second": samples_per_sec,
                "Mean latency (seconds)": mean_latency_sec
            }
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    timestamp = datetime.now(timezone.utc)
    unix_time = timestamp.timestamp()

    # Insert into MongoDB
    result_doc = {
        "timestamp": timestamp.isoformat(),
        "unix_time": unix_time,
        "benchmark_results": benchmark_results
    }

    result = db.benchmark_results.insert_one(result_doc)
    # print(f"Inserted benchmark results with _id: {result.inserted_id}")
