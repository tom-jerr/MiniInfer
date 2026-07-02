"""MiniInfer benchmark — 方法论对齐 nano-vllm/benchmarks/benchmark_vllm.py。

随机 token ids、256 seqs、输入 100-1024、输出 100-1024、temperature=0.6、ignore_eos。
warmup 后计时，throughput = 总输出 token / 时间。
"""
import argparse
import os
import time
from random import randint, seed

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="/root/models/Qwen3-0.6B")
  p.add_argument("--num-seqs", type=int, default=256)
  p.add_argument("--max-input-len", type=int, default=1024)
  p.add_argument("--max-output-len", type=int, default=1024)
  p.add_argument("--max-num-seqs", type=int, default=256)
  p.add_argument("--max-model-len", type=int, default=4096)
  p.add_argument("--page-size", type=int, default=256)
  p.add_argument("--attn-backend", default="flash_attn")
  p.add_argument("--eager", action="store_true")
  args = p.parse_args()

  seed(0)
  prompt_token_ids = [
    [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
    for _ in range(args.num_seqs)
  ]
  sampling_params = [
    SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
    for _ in range(args.num_seqs)
  ]

  engine = LLMEngine(
    model=args.model,
    use_multiprocess=False,
    max_num_seqs=args.max_num_seqs,
    max_model_len=args.max_model_len,
    attention_backend=args.attn_backend,
    page_size=args.page_size,
    enforce_eager=args.eager,
  )
  engine.start()

  # warmup
  engine.generate(prompts=["Benchmark: "], sampling_params=SamplingParams(), use_tqdm=False)

  t0 = time.time()
  outs = engine.generate(prompts=prompt_token_ids, sampling_params=sampling_params, use_tqdm=False)
  dt = time.time() - t0
  engine.stop()

  total_tokens = sum(sp.max_tokens for sp in sampling_params)
  print(f"[MiniInfer] Total: {total_tokens}tok, Time: {dt:.2f}s, Throughput: {total_tokens/dt:.2f}tok/s")


if __name__ == "__main__":
  main()
