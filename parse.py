import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from datetime import datetime, timezone
import glob
from dotenv import load_dotenv
import re
import json
import uuid
import subprocess

# Load environment variables
load_dotenv()

class PostgreSQLBenchmarkDB:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(
                host=os.getenv('DB_HOST'),
                port=os.getenv('DB_PORT', '5432'),
                database=os.getenv('DB_NAME'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASSWORD')
            )
            print("Successfully connected to PostgreSQL database")
        except Exception as e:
            print(f"Database connection error: {e}")
            raise
    
    def get_gpu_info(self):
        """Get current GPU information"""
        try:
            # Get GPU model and ID using nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for i, line in enumerate(lines):
                    if line.strip():
                        name, uuid_gpu = line.split(', ')
                        gpu_info.append({
                            'gpu_id': str(i),
                            'gpu_model': name.strip(),
                            'gpu_uuid': uuid_gpu.strip()
                        })
                return gpu_info
            else:
                # Fallback to basic info
                return [{'gpu_id': '0', 'gpu_model': 'Unknown GPU', 'gpu_uuid': 'unknown'}]
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return [{'gpu_id': '0', 'gpu_model': 'Unknown GPU', 'gpu_uuid': 'unknown'}]
    
    def get_hostname(self):
        """Get current hostname"""
        try:
            return subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
        except:
            return "unknown-host"
    
    def insert_benchmark_result(self, gpu_node, gpu_id, gpu_model, benchmark_type, benchmark_data):
        """Insert raw benchmark result into database"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO raw_benchmark_results 
                    (gpu_node, gpu_id, gpu_model, benchmark_type, benchmark_data, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING audit_id
                """, (gpu_node, gpu_id, gpu_model, benchmark_type, Json(benchmark_data), datetime.now(timezone.utc)))
                
                audit_id = cursor.fetchone()[0]
                self.connection.commit()
                print(f"Inserted benchmark result with audit_id: {audit_id}")
                return audit_id
        except Exception as e:
            print(f"Error inserting benchmark result: {e}")
            self.connection.rollback()
            raise
    
    def update_gpu_aggregates(self, gpu_node, gpu_id, gpu_model, benchmark_type, metrics):
        """Update GPU aggregates with new metrics"""
        try:
            with self.connection.cursor() as cursor:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        cursor.execute("""
                            SELECT update_gpu_aggregates(%s, %s, %s, %s, %s, %s)
                        """, (gpu_node, gpu_id, gpu_model, benchmark_type, metric_name, float(value)))
            
            self.connection.commit()
            print(f"Updated GPU aggregates for {gpu_node}:{gpu_id}")
        except Exception as e:
            print(f"Error updating GPU aggregates: {e}")
            self.connection.rollback()
            raise
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

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

def parse_gpu_status(file_path):
    with open(file_path, 'r') as f:
            lines = f.readlines()
    gpu_status = {}
    pattern = re.compile(r'GPU\s+(\d+):\s*(OK|FAIL|ERROR|.*)', re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match:
            gpu_id = match.group(1)
            status = match.group(2).strip().upper()
            gpu_status[gpu_id] = status
    return gpu_status

def parse_hpl_output(file_path: str):
    results = {
        "accuracy": {},
        "performance": {}
    }

    with open(file_path, 'r') as file:
        content = file.read()

    # Extract main test result
    final_perf_match = re.search(r'WC0\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.eE+-]+)', content)
    if final_perf_match:
        results["performance"] = {
            "N": int(final_perf_match.group(1)),
            "NB": int(final_perf_match.group(2)),
            "P": int(final_perf_match.group(3)),
            "Q": int(final_perf_match.group(4)),
            "time_sec": float(final_perf_match.group(5)),
            "gflops": float(final_perf_match.group(6))
        }

    # Extract residuals
    residual_match = re.search(r'\|\|Ax-b\|\|_oo.*?=\s+([\deE.+-]+).*?PASSED.*?\|\|A\|\|_oo.*?=\s+([\deE.+-]+).*?\|\|x\|\|_oo.*?=\s+([\deE.+-]+).*?\|\|b\|\|_oo.*?=\s+([\deE.+-]+)', content, re.DOTALL)
    if residual_match:
        results["accuracy"] = {
            "residual_ratio": float(residual_match.group(1)),
            "A_norm": float(residual_match.group(2)),
            "x_norm": float(residual_match.group(3)),
            "b_norm": float(residual_match.group(4)),
            "passed": True
        }

    return results

def parse_hpcg_output(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    results = {}

    # Grid and domain info
    grid_info = re.search(r"Process Grid: (\d+)x(\d+)x(\d+)", content)
    domain_info = re.search(r"Local Domain: (\d+)x(\d+)x(\d+)", content)
    if grid_info and domain_info:
        results['process_grid'] = tuple(map(int, grid_info.groups()))
        results['local_domain'] = tuple(map(int, domain_info.groups()))

    # Iteration summary
    iteration_summary = re.search(r"Iteration Count Information::Total number of reference iterations=(\d+).*?Total number of optimized iterations=(\d+)", content, re.DOTALL)
    if iteration_summary:
        results['reference_iterations'] = int(iteration_summary.group(1))
        results['optimized_iterations'] = int(iteration_summary.group(2))

    # Memory usage
    mem_usage = re.search(r"Total memory used for data \(Gbytes\)=(\d+\.\d+)", content)
    if mem_usage:
        results['memory_used_GB'] = float(mem_usage.group(1))

    # Performance summary
    perf_summary = re.search(
        r"GFLOP/s Summary::Raw Total=(\d+\.\d+).*?Total with convergence overhead=(\d+\.\d+).*?overhead=.*?=\s*(\d+\.\d+)",
        content,
        re.DOTALL,
    )
    if perf_summary:
        results['gflops_raw_total'] = float(perf_summary.group(1))
        results['gflops_with_convergence'] = float(perf_summary.group(2))
        results['gflops_final'] = float(perf_summary.group(3))

    # Final validation
    final_validation = re.search(r"Final Summary::HPCG result is VALID.*?GFLOP/s rating of=(\d+\.\d+)", content)
    if final_validation:
        results['hpcg_valid'] = True
        results['hpcg_rating'] = float(final_validation.group(1))
    else:
        results['hpcg_valid'] = False

    return results

def parse_stream_output(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    result = {
        "bus_width_bits": None,
        "peak_bandwidth_gbps": None,
        "array_size_mb": None,
        "tests": {}
    }

    # Match device info
    device_match = re.search(
        r'Device 0: "([^"]+)"\s+\d+\s+SMs.*?Memory:\s+(\d+)MHz x (\d+)-bit\s+=\s+([\d.]+)\s+GB/s',
        text,
        re.DOTALL
    )
    if device_match:
        result["device_name"] = device_match.group(1)
        result["bus_width_bits"] = int(device_match.group(3))
        result["peak_bandwidth_gbps"] = float(device_match.group(4))

    # Match array size
    array_match = re.search(r'Array size \(double\)=.*\(([\d.]+)\s+MB\)', text)
    if array_match:
        result["array_size_mb"] = float(array_match.group(1))

    # Match tests (Copy, Scale, Add, Triad) using MULTILINE
    test_pattern = re.compile(
        r'^\s*(Copy|Scale|Add|Triad):\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
        re.MULTILINE
    )
    for match in test_pattern.finditer(text):
        name = match.group(1)
        result["tests"][name] = {
            "rate_MBps": float(match.group(2)),
            "avg_time_sec": float(match.group(3)),
            "min_time_sec": float(match.group(4)),
            "max_time_sec": float(match.group(5))
        }

    return result

def main():
    # Initialize database connection
    db = PostgreSQLBenchmarkDB()
    
    # Get system information
    gpu_info_list = db.get_gpu_info()
    hostname = db.get_hostname()
    
    print(f"Processing benchmarks for host: {hostname}")
    print(f"Found {len(gpu_info_list)} GPU(s)")
    
    # Process MLPerf results
    try:
        test_results_dirs = find_all_test_result_paths()
        models_list = ["bert-99"]
        
        for model_name in models_list:
            try:
                model_dirs = [os.path.join(test_result_dir, model_name) for test_result_dir in test_results_dirs]
                summary_file = find_latest_run(model_dirs)
                samples_per_sec, mean_latency_sec = parse_mlperf_result(summary_file, model_name)
                
                # Store results for each GPU
                for gpu_info in gpu_info_list:
                    benchmark_data = {
                        "model": model_name,
                        "samples_per_second": samples_per_sec,
                        "mean_latency_seconds": mean_latency_sec,
                        "mean_latency_ns": mean_latency_sec * 1e9
                    }
                    
                    # Insert raw result
                    audit_id = db.insert_benchmark_result(
                        hostname, 
                        gpu_info['gpu_id'], 
                        gpu_info['gpu_model'], 
                        'mlperf', 
                        benchmark_data
                    )
                    
                    # Update aggregates
                    metrics = {
                        'samples_per_second': samples_per_sec,
                        'mean_latency_seconds': mean_latency_sec
                    }
                    db.update_gpu_aggregates(hostname, gpu_info['gpu_id'], gpu_info['gpu_model'], 'mlperf', metrics)
                    
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
    
    except Exception as e:
        print(f"Error processing MLPerf results: {e}")
    
    # Process GPU burn results
    try:
        if os.path.exists("gpu_burn.txt"):
            gpu_status = parse_gpu_status("gpu_burn.txt")
            
            for gpu_id, status in gpu_status.items():
                benchmark_data = {
                    "status": status,
                    "test_type": "stress_test"
                }
                
                # Find corresponding GPU info
                gpu_info = next((gpu for gpu in gpu_info_list if gpu['gpu_id'] == gpu_id), 
                               {'gpu_id': gpu_id, 'gpu_model': 'Unknown GPU'})
                
                # Insert raw result
                audit_id = db.insert_benchmark_result(
                    hostname, 
                    gpu_info['gpu_id'], 
                    gpu_info['gpu_model'], 
                    'gpu_burn', 
                    benchmark_data
                )
                
                # Update aggregates (reliability score)
                reliability_score = 1.0 if status == 'OK' else 0.0
                metrics = {'reliability_score': reliability_score}
                db.update_gpu_aggregates(hostname, gpu_info['gpu_id'], gpu_info['gpu_model'], 'gpu_burn', metrics)
    
    except Exception as e:
        print(f"Error processing GPU burn results: {e}")
    
    # Process HPC results
    try:
        # HPL results
        if os.path.exists("hpl_results.txt"):
            hpl_results = parse_hpl_output("hpl_results.txt")
            
            for gpu_info in gpu_info_list:
                benchmark_data = {
                    "hpl_results": hpl_results
                }
                
                audit_id = db.insert_benchmark_result(
                    hostname, 
                    gpu_info['gpu_id'], 
                    gpu_info['gpu_model'], 
                    'hpl', 
                    benchmark_data
                )
                
                # Update aggregates with HPL performance
                if 'performance' in hpl_results and 'gflops' in hpl_results['performance']:
                    metrics = {'gflops': hpl_results['performance']['gflops']}
                    db.update_gpu_aggregates(hostname, gpu_info['gpu_id'], gpu_info['gpu_model'], 'hpl', metrics)
        
        # HPCG results
        if os.path.exists("hpcg_results.txt"):
            hpcg_results = parse_hpcg_output("hpcg_results.txt")
            
            for gpu_info in gpu_info_list:
                benchmark_data = {
                    "hpcg_results": hpcg_results
                }
                
                audit_id = db.insert_benchmark_result(
                    hostname, 
                    gpu_info['gpu_id'], 
                    gpu_info['gpu_model'], 
                    'hpcg', 
                    benchmark_data
                )
                
                # Update aggregates with HPCG performance
                if 'hpcg_rating' in hpcg_results:
                    metrics = {'hpcg_rating': hpcg_results['hpcg_rating']}
                    db.update_gpu_aggregates(hostname, gpu_info['gpu_id'], gpu_info['gpu_model'], 'hpcg', metrics)
        
        # STREAM results
        if os.path.exists("stream_results.txt"):
            stream_results = parse_stream_output("stream_results.txt")
            
            for gpu_info in gpu_info_list:
                benchmark_data = {
                    "stream_results": stream_results
                }
                
                audit_id = db.insert_benchmark_result(
                    hostname, 
                    gpu_info['gpu_id'], 
                    gpu_info['gpu_model'], 
                    'stream', 
                    benchmark_data
                )
                
                # Update aggregates with STREAM performance
                if 'tests' in stream_results:
                    for test_name, test_data in stream_results['tests'].items():
                        metrics = {f'{test_name.lower()}_bandwidth_mbps': test_data['rate_MBps']}
                        db.update_gpu_aggregates(hostname, gpu_info['gpu_id'], gpu_info['gpu_model'], 'stream', metrics)
    
    except Exception as e:
        print(f"Error processing HPC results: {e}")
    
    # Close database connection
    db.close()
    
    print("Benchmark processing completed successfully!")

if __name__ == "__main__":
    main()

