import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM
import sys
import os
from torch.profiler import profile, record_function, ProfilerActivity

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.fused_qwen2 import Qwen2Model


def main():
  parser = argparse.ArgumentParser(description="Profile Qwen2 Model (No Cache)")
  parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
  parser.add_argument("--seq-len", type=int, default=128)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--iter", type=int, default=20)
  parser.add_argument("--profile", action="store_true", help="Enable torch profiler")
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Loading {args.model} on {device}...")

  hf_model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.float16, device_map=device
  )
  model = Qwen2Model(hf_model)
  # compile
  torch.compile(model, mode="max-autotune")
  model.eval()

  # Dummy input
  input_ids = torch.randint(0, hf_model.config.vocab_size, (1, args.seq_len), device=device)

  # Warmup
  print(f"Warming up ({args.warmup} iters)...")
  for _ in range(args.warmup):
    with torch.no_grad():
      model(input_ids, use_cache=False)
  torch.cuda.synchronize()

  # Benchmark Latency
  print(f"Benchmarking ({args.iter} iters)...")
  latencies = []
  for _ in range(args.iter):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
      model(input_ids, use_cache=False)
    end.record()
    torch.cuda.synchronize()
    latencies.append(start.elapsed_time(end))  # ms

  avg_latency = np.mean(latencies)
  std_latency = np.std(latencies)

  # Bandwidth Calculation
  # Model Size (Weights)
  param_count = sum(p.numel() for p in model.parameters())
  model_size_bytes = param_count * 2  # fp16

  # IO Size (Input + Output)
  # Output: 1 * seq_len * vocab_size * 2 bytes (logits)
  output_size_bytes = 1 * args.seq_len * hf_model.config.vocab_size * 2

  # Total Memory Access (Approximate: Weights + Output Write)
  # Note: This ignores intermediate activations, which is significant for training but less for inference.
  # For memory bound inference, weights are the main bottleneck.
  total_bytes = model_size_bytes + output_size_bytes
  bandwidth = (total_bytes / 1e9) / (avg_latency / 1000)  # GB/s

  # Results Table
  results = {
    "Metric": [
      "Sequence Length",
      "Avg Latency (ms)",
      "Std Latency (ms)",
      "Model Size (GB)",
      "Est. Bandwidth (GB/s)",
    ],
    "Value": [
      args.seq_len,
      f"{avg_latency:.3f}",
      f"{std_latency:.3f}",
      f"{model_size_bytes/1e9:.3f}",
      f"{bandwidth:.2f}",
    ],
  }
  df = pd.DataFrame(results)
  print("\n" + "=" * 40)
  print("Benchmark Results (No Cache)")
  print("=" * 40)
  print(df.to_string(index=False))

  # Profiling
  if args.profile:
    print("\nRunning Torch Profiler...")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
      with record_function("model_forward"):
        with torch.no_grad():
          model(input_ids, use_cache=False)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("qwen2_profile.json")
    print("Trace saved to qwen2_profile.json")


if __name__ == "__main__":
  main()
