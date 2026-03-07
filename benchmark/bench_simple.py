import argparse
import os
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


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


def main() -> None:
  parser = argparse.ArgumentParser(description="MiniInfer simple benchmark + torch.profiler trace export.")
  parser.add_argument("--model", default=os.getenv("MINIINFER_MODEL", "Qwen/Qwen2-0.5B-Instruct"))
  parser.add_argument("--batch-size", type=int, default=int(os.getenv("MINIINFER_BS", "8")))
  parser.add_argument("--prompt", default=os.getenv("MINIINFER_PROMPT", "Hello"))
  parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MINIINFER_MAX_TOKENS", "64")))
  parser.add_argument("--temperature", type=float, default=float(os.getenv("MINIINFER_TEMP", "0.7")))
  parser.add_argument(
    "--trace",
    default=os.getenv("MINIINFER_TRACE", "engine_profile.json"),
    help="Chrome trace output path (set empty string to disable).",
  )
  parser.add_argument(
    "--full-trace",
    action="store_true",
    default=_env_flag("MINIINFER_FULL_TRACE", False),
    help="Enable with_stack + record_shapes + profile_memory (much slower, huge JSON).",
  )
  args = parser.parse_args()

  llm = LLMEngine(model=args.model, enforce_eager=False)
  try:
    prompts = [args.prompt] * int(args.batch_size)
    params = [
      SamplingParams(max_tokens=int(args.max_tokens), temperature=float(args.temperature))
      for _ in prompts
    ]

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
      activities.append(ProfilerActivity.CUDA)

    with profile(
      activities=activities,
      record_shapes=bool(args.full_trace),
      profile_memory=bool(args.full_trace),
      with_stack=bool(args.full_trace),
    ) as prof:
      llm.generate(prompts, params, use_tqdm=False)

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=80))
    if torch.cuda.is_available():
      print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))

    trace_path = str(args.trace or "").strip()
    if trace_path:
      trace_file = Path(trace_path)
      trace_file.parent.mkdir(parents=True, exist_ok=True)
      print(
        f"exporting chrome trace to {trace_file} (this may take a while; use --full-trace for stacks/shapes/memory)...",
        flush=True,
      )
      t0 = perf_counter()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      prof.export_chrome_trace(str(trace_file))
      dt = perf_counter() - t0
      size_mb = trace_file.stat().st_size / (1024 * 1024) if trace_file.exists() else 0.0
      print(f"saved: {trace_file} ({size_mb:.1f} MB, {dt:.2f}s)")
  finally:
    llm.stop()


if __name__ == "__main__":
  main()
