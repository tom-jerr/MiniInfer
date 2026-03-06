# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import argparse
import os
import time
from random import randint, seed
import torch
from miniinfer.utils.sampling_params import SamplingParams
from miniinfer.engine.llm_engine import LLMEngine

import logging

logging.basicConfig(
  level=logging.ERROR,
  format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
  force=True,  # Python 3.8+：确保生效（避免被别的库提前配置）
)

# 默认模型路径映射
MODEL_SHORTCUTS = {
  "qwen2-0.5b": "Qwen/Qwen2-0.5B-Instruct",
  "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}


def main():
  parser = argparse.ArgumentParser(description="MiniInfer throughput benchmark")
  parser.add_argument(
    "--model",
    type=str,
    default="qwen2-0.5b",
    help="Model shortcut (qwen2-0.5b, qwen3-0.6b) or full path/HF hub ID",
  )
  parser.add_argument("--num-seqs", type=int, default=256)
  parser.add_argument("--max-input-len", type=int, default=1024)
  parser.add_argument("--max-output-len", type=int, default=1024)
  args = parser.parse_args()

  model_path = MODEL_SHORTCUTS.get(args.model.lower(), args.model)

  seed(0)
  num_seqs = args.num_seqs
  max_input_len = args.max_input_len
  max_ouput_len = args.max_output_len

  # align the hyperparameters
  llm = LLMEngine(
    model=model_path,
    enable_chunked_prefill=False,
  )

  prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
  ]
  sampling_params = [
    SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
    for _ in range(num_seqs)
  ]
  llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
  # activities = [ProfilerActivity.CPU]
  # if torch.cuda.is_available():
  #   activities.append(ProfilerActivity.CUDA)

  # with profile(
  #   activities=activities,
  #   record_shapes=True,
  #   profile_memory=True,
  #   with_stack=True,
  # ) as prof:
  t = time.time()
  llm.generate(prompt_token_ids, sampling_params)
  t = time.time() - t
  total_tokens = sum(sp.max_tokens for sp in sampling_params)
  throughput = total_tokens / t
  print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
  # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=80))
  # if torch.cuda.is_available():
  #   print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))

  # prof.export_chrome_trace("engine_profile.json")
  # print("saved: engine_profile.json")


if __name__ == "__main__":
  main()
