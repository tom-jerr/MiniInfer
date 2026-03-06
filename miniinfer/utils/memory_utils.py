from __future__ import annotations

import logging
from typing import Optional

import torch

from miniinfer.config.engine.config import EngineConfig

logger = logging.getLogger(__name__)


def get_available_gpu_memory(
  *,
  gpu_memory_utilization: float,
  device: Optional[int] = None,
  log: Optional[logging.Logger] = None,
) -> int:
  """
  Get available GPU memory in bytes.

  Uses `torch.cuda.mem_get_info` to account for CUDA context, driver overhead,
  other processes, and PyTorch allocations.
  """
  if not torch.cuda.is_available():
    return 0

  dev = torch.cuda.current_device() if device is None else int(device)
  free_memory, total_memory = torch.cuda.mem_get_info(dev)

  used_memory = total_memory - free_memory
  available_memory = int(total_memory * float(gpu_memory_utilization)) - used_memory

  (log or logger).info(
    "GPU Memory - Total: %.2fGB, Used: %.2fGB, Free: %.2fGB, "
    "Available for KV Cache (with %.1f%% utilization): %.2fGB",
    total_memory / (1024**3),
    used_memory / (1024**3),
    free_memory / (1024**3),
    float(gpu_memory_utilization) * 100.0,
    available_memory / (1024**3),
  )

  return max(0, int(available_memory))


def get_kv_cache_bytes_per_token(
  config: EngineConfig,
  *,
  log: Optional[logging.Logger] = None,
) -> int:
  """
  Compute KV cache bytes per token.

  KV cache size = 2 (K + V) × num_layers × num_kv_heads × head_dim × dtype_size
  """
  dtype_size = torch.tensor([], dtype=config.dtype).element_size()
  head_dim = config.hf_config.hidden_size // config.hf_config.num_attention_heads
  bytes_per_token = (
    2
    * config.hf_config.num_hidden_layers
    * config.hf_config.num_key_value_heads
    * head_dim
    * dtype_size
  )

  (log or logger).info(
    "KV Cache per token: %d bytes (layers=%d, heads=%d, head_dim=%d, dtype=%s)",
    bytes_per_token,
    config.hf_config.num_hidden_layers,
    config.hf_config.num_key_value_heads,
    head_dim,
    config.dtype,
  )

  return int(bytes_per_token)


def calc_max_total_tokens(
  config: EngineConfig,
  *,
  reserved_ratio: float = 0.20,
  device: Optional[int] = None,
  gpu_memory_utilization: Optional[float] = None,
  log: Optional[logging.Logger] = None,
) -> int:
  """
  Calculate the maximum number of tokens the KV cache can hold.

  Notes:
  - Intended to be called after the model is loaded, using actual remaining VRAM.
  - Aligns to `page_size` to match paged attention layout assumptions.
  """
  util = (
    float(config.gpu_memory_utilization)
    if gpu_memory_utilization is None
    else float(gpu_memory_utilization)
  )
  available_memory = get_available_gpu_memory(gpu_memory_utilization=util, device=device, log=log)

  bytes_per_token = get_kv_cache_bytes_per_token(config, log=log)
  if bytes_per_token <= 0:
    (log or logger).error("bytes_per_token is 0, cannot calculate max_total_tokens")
    return 0

  usable_memory = int(available_memory * (1.0 - float(reserved_ratio)))

  page_size = int(config.page_size)
  raw_tokens = usable_memory // bytes_per_token
  max_total_tokens = raw_tokens - page_size
  max_total_tokens = (max_total_tokens // page_size) * page_size

  min_kv_tokens = 2 * page_size
  max_total_tokens = max(max_total_tokens, min_kv_tokens)

  if max_total_tokens <= min_kv_tokens:
    (log or logger).warning(
      "KV cache capacity (%d tokens) is at minimum. "
      "Available GPU memory may be insufficient. "
      "Consider using a smaller model or increasing GPU memory.",
      max_total_tokens,
    )

  (log or logger).info(
    "Calculated max_total_tokens: %d (available: %.2fGB, usable: %.2fGB, "
    "bytes_per_token: %d, raw_tokens: %d, page_size: %d, min_kv_tokens: %d, "
    "actual_allocation: %.2fGB)",
    max_total_tokens,
    available_memory / (1024**3),
    usable_memory / (1024**3),
    bytes_per_token,
    raw_tokens,
    page_size,
    min_kv_tokens,
    ((max_total_tokens + page_size) * bytes_per_token) / (1024**3),
  )

  return int(max_total_tokens)
