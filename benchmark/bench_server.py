# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import argparse
import os
import time
from pathlib import Path
from random import randint, seed

import torch
from torch.profiler import ProfilerActivity, profile

from miniinfer.utils.sampling_params import SamplingParams
from miniinfer.engine.llm_engine import LLMEngine

import logging

logging.basicConfig(
  level=logging.WARNING,
  format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
  force=True,  # Python 3.8+：确保生效（避免被别的库提前配置）
)

# 默认模型路径映射
MODEL_SHORTCUTS = {
  "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",
  "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def _env_flag(name: str, default: bool) -> bool:
  value = os.getenv(name)
  if value is None:
    return default
  value = value.strip().lower()
  if value in {"1", "true", "yes", "on"}:
    return True
  if value in {"0", "false", "no", "off"}:
    return False
  return default


def main():
  parser = argparse.ArgumentParser(
    description="MiniInfer throughput benchmark (+ optional torch.profiler trace export)."
  )
  parser.add_argument(
    "--model",
    type=str,
    default="qwen2-0.5b",
    help="Model shortcut (qwen2-0.5b, qwen3-0.6b) or full path/HF hub ID",
  )
  parser.add_argument("--num-seqs", type=int, default=256)
  parser.add_argument("--max-input-len", type=int, default=1024)
  parser.add_argument("--max-output-len", type=int, default=1024)
  parser.add_argument(
    "--overlap",
    action=argparse.BooleanOptionalAction,
    default=_env_flag("MINIINFER_OVERLAP", True),
    help="Enable overlap scheduling (set --no-overlap to disable).",
  )
  parser.add_argument(
    "--profile",
    action="store_true",
    default=_env_flag("MINIINFER_PROFILE", False),
    help="Enable torch.profiler for the main generate() call.",
  )
  parser.add_argument(
    "--trace",
    default=os.getenv("MINIINFER_TRACE", ""),
    help="Chrome trace output path (set empty string to disable export). Implies --profile.",
  )
  parser.add_argument(
    "--full-trace",
    action="store_true",
    default=_env_flag("MINIINFER_FULL_TRACE", False),
    help="Enable with_stack + record_shapes + profile_memory (much slower, huge JSON).",
  )
  args = parser.parse_args()

  model_path = MODEL_SHORTCUTS.get(args.model.lower(), args.model)

  seed(0)
  num_seqs = args.num_seqs
  max_input_len = args.max_input_len
  max_output_len = args.max_output_len

  # align the hyperparameters
  llm = LLMEngine(
    model=model_path,
    enable_chunked_prefill=False,
    enable_overlap=bool(args.overlap),
  )

  try:
    prompt_token_ids = [
      [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    sampling_params = [
      SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
      for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    if torch.cuda.is_available():
      torch.cuda.synchronize()

    trace_path = str(args.trace or "").strip()
    enable_profile = bool(args.profile) or bool(trace_path)

    if enable_profile:
      activities = [ProfilerActivity.CPU]
      if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

      with profile(
        activities=activities,
        record_shapes=bool(args.full_trace),
        profile_memory=bool(args.full_trace),
        with_stack=bool(args.full_trace),
      ) as prof:
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params)
        if torch.cuda.is_available():
          torch.cuda.synchronize()
        t = time.time() - t

      total_tokens = sum(sp.max_tokens for sp in sampling_params)
      throughput = total_tokens / t
      print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")

      print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=80))
      if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))

      if trace_path:
        trace_file = Path(trace_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        print(
          f"exporting chrome trace to {trace_file} (this may take a while; use --full-trace for stacks/shapes/memory)...",
          flush=True,
        )
        t0 = time.perf_counter()
        if torch.cuda.is_available():
          torch.cuda.synchronize()
        prof.export_chrome_trace(str(trace_file))
        dt = time.perf_counter() - t0
        size_mb = trace_file.stat().st_size / (1024 * 1024) if trace_file.exists() else 0.0
        print(f"saved: {trace_file} ({size_mb:.1f} MB, {dt:.2f}s)")
    else:
      t = time.time()
      llm.generate(prompt_token_ids, sampling_params)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      t = time.time() - t
      total_tokens = sum(sp.max_tokens for sp in sampling_params)
      throughput = total_tokens / t
      print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
  finally:
    llm.stop()


if __name__ == "__main__":
  main()
