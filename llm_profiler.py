# Run training or inference on a model in PyTorch
# Profile each step using torch.profiler or nvprof
# Set internal CUDA properties
# Grab the model, send to device - HuggingFace
# Records GPU & CPU kernel-level activity
# Profile each stage of training

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler, record_function
import torch.nn as nn
import torch.optim as optim
import subprocess
import time
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Uses nvidia-smi to get VRAM, power, and temperature
def get_gpu_metrics(device_index=0):
    """Returns memory (GB), power (W), temperature (C)"""
    mem_reserved = torch.cuda.memory_reserved(device_index) / 1024**3 # in GB

    try:
        smi_output = subprocess.check_output([
            "nvidia-smi",
            f"--query-gpu=power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]).decode().strip().split("\n")[device_index]

        power, temp = map(float, smi_output.strip().split(','))
    except Exception:
        power, temp = 0.0, 0.0

    return round(mem_reserved, 2), round(power, 2), round(temp, 2)

# Logs GPU metrics for each stage of training or inference
def log_stage(stage_name, device_index=0):
    mem, power, temp = get_gpu_metrics(device_index)
    print(f"[{stage_name}] VRAM: {mem:.2f} GB | Power: {power:.2f} W | Temp: {temp:.1f} °C")

def run_hf_model_profile(model_name="bert-base-uncased", batch_size=4, seq_len=128,
                         dtype=torch.float32, mode="train", use_compile=False):

    # Set internal CUDA properties
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # TODO look into more PyTorch exposed CUDA internal variables 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True # Forces deterministic behavior
    torch.backends.cudnn.allow_tf32 = False # No tensor cores used
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False # Precision reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision('highest')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    model.train() if mode == "train" else model.eval()

    # Dummy input
    dummy_input = ["The quick brown fox jumps over the lazy dog."] * batch_size
    tokens = tokenizer(dummy_input, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in tokens.items()}
    labels = torch.randint(0, 2, (batch_size,)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # Profiler - keep track of kernels launched
    trace_dir = f"./runs/{model_name}_{mode}/plugins/profile/{int(time.time())}"
    os.makedirs(trace_dir, exist_ok=True)

    # List to store GPU metrics per stage
    gpu_metrics_log = []
    stage_names_log = []

    def log_stage(stage_name, device_index=0):
        mem, power, temp = get_gpu_metrics(device_index)
        print(f"[{stage_name}] VRAM: {mem:.2f} GB | Power: {power:.2f} W | Temp: {temp:.1f} °C")
        gpu_metrics_log.append({'stage': stage_name, 'vram_gb': mem, 'power_w': power, 'temp_c': temp})
        stage_names_log.append(stage_name)

    with profile( # Uses torch.profiler to track CPU and GPU activity
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=4),
        on_trace_ready=tensorboard_trace_handler(trace_dir), 
        record_shapes=True,
        with_stack=True,  # Enable stack traces for better debugging
        with_flops=True,
        profile_memory=True,  # Enable memory profiling
    ) as prof:

        for epoch in range(6):  # 1 wait + 1 warmup + 4 active = 6
            print(f"\n[Epoch {epoch+1}/6]")

            if mode == "train":
                model.train()
                optimizer.zero_grad()

                with record_function("forward"):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                log_stage("Forward")

                with record_function("loss_backward"):
                    loss.backward()
                log_stage("Backward")

                with record_function("optimizer_step"):
                    optimizer.step()
                log_stage("Optimizer Step")

            elif mode == "infer":  # Inference
                model.eval()
                with torch.inference_mode(), record_function("inference_forward"):
                    _ = model(**inputs)
                log_stage("Inference Forward")

            prof.step()

    print("Done. TensorBoard trace saved in ./runs/")
    print(f"Trace directory: {os.path.abspath(trace_dir)}")
    print(f"Trace files: {os.listdir(trace_dir) if os.path.exists(trace_dir) else 'Directory not found'}")
    
    # Print profiler statistics
    print("\nProfiler Statistics:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # --- Save and plot GPU metrics ---
    gpu_df = pd.DataFrame(gpu_metrics_log)
    gpu_csv_path = os.path.join(trace_dir, "gpu_metrics.csv")
    gpu_df.to_csv(gpu_csv_path, index=False)

    plt.figure(figsize=(10, 6))
    for col in ['vram_gb', 'power_w', 'temp_c']:
        plt.plot(gpu_df['stage'], gpu_df[col], marker='o', label=col)
    plt.xlabel('Stage')
    plt.ylabel('Value')
    plt.title('GPU Metrics per Stage')
    plt.legend()
    plt.tight_layout()
    gpu_plot_path = os.path.join(trace_dir, "gpu_metrics.png")
    plt.savefig(gpu_plot_path)
    plt.close()
    print(f"Saved GPU metrics plot to {gpu_plot_path}")

    # --- Save and plot kernel stats ---
    kernel_stats = prof.key_averages()
    kernel_data = []
    for item in kernel_stats:
        kernel_data.append({
            'name': item.key,
            'cpu_time_total': getattr(item, 'cpu_time_total', 0),
            'cuda_time_total': getattr(item, 'cuda_time_total', 0),
            'self_cuda_time_total': getattr(item, 'self_cuda_time_total', 0),
            'cpu_memory_usage': getattr(item, 'cpu_memory_usage', 0),
            'cuda_memory_usage': getattr(item, 'cuda_memory_usage', 0),
            'flops': getattr(item, 'flops', None)
        })
    kernel_df = pd.DataFrame(kernel_data)
    kernel_csv_path = os.path.join(trace_dir, "kernel_stats.csv")
    kernel_df.to_csv(kernel_csv_path, index=False)

    # Plot top 10 kernels by CUDA time
    top_kernels = kernel_df.sort_values('cuda_time_total', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    plt.barh(top_kernels['name'], top_kernels['cuda_time_total'])
    plt.xlabel('CUDA Time Total (us)')
    plt.title('Top 10 Kernels by CUDA Time')
    plt.tight_layout()
    kernel_plot_path = os.path.join(trace_dir, "kernel_stats.png")
    plt.savefig(kernel_plot_path)
    plt.close()
    print(f"Saved kernel stats plot to {kernel_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM Profiler with TensorBoard integration')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], 
                       help='Mode: train or infer')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='Model name from HuggingFace')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Sequence length')
    parser.add_argument('--use_compile', action='store_true',
                       help='Use torch.compile')
    
    args = parser.parse_args()
    
    run_hf_model_profile(
        model_name=args.model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dtype=torch.float32,
        mode=args.mode,
        use_compile=args.use_compile
    )

