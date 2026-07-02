"""MiniInfer profiler: 跑一个有代表性的 decode-heavy 负载，导出 chrome trace +
打印 top GPU kernel / CPU op 表，定位优化点。"""
import os
import time
from random import randint, seed

import torch
from torch.profiler import ProfilerActivity, profile

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


def main():
  seed(0)
  num_seqs = int(os.getenv("P_SEQS", "32"))
  out_len = int(os.getenv("P_OUT", "128"))
  in_len = int(os.getenv("P_IN", "256"))

  prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(100, in_len))] for _ in range(num_seqs)
  ]
  sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len) for _ in range(num_seqs)]

  engine = LLMEngine(
    model="/root/models/Qwen3-0.6B", use_multiprocess=False,
    max_num_seqs=num_seqs, max_model_len=4096, attention_backend="flash_attn",
  )
  engine.start()
  engine.generate(prompts=["warmup"], sampling_params=SamplingParams(max_tokens=4), use_tqdm=False)

  # profiled run
  with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False, with_stack=False, profile_memory=False,
  ) as prof:
    t0 = time.time()
    engine.generate(prompts=prompt_token_ids, sampling_params=sp, use_tqdm=False)
    dt = time.time() - t0

  total = sum(s.max_tokens for s in sp)
  print(f"[profiled] {total}tok in {dt:.2f}s = {total/dt:.0f} tok/s")

  prof.export_chrome_trace("/tmp/miniinfer_trace.json")
  print("\n===== top CUDA kernels (self time) =====")
  print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
  print("\n===== top CPU ops (self time) =====")
  print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
  engine.stop()


if __name__ == "__main__":
  main()
