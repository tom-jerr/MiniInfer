import torch
from typing import Callable
from .linear import softmax
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SamplingBatchInfo:
  temperature: torch.Tensor
  top_ps: torch.Tensor
  top_ks: torch.Tensor
  # Grammar-based sampling
  vocab_size: int
  # Optional CPU-side maximum top-k used to avoid GPU->CPU sync in batched top-k.
  # If not provided, batched_sample may fall back to a sync to compute it.
  max_top_k: Optional[int] = None
  grammars: Optional[List] = None
  vocab_mask: Optional[torch.Tensor] = None
  apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

  has_custom_logits_processor: bool = False


class Sampler(nn.Module):
  def __init__(self):
    super().__init__()

  def _preprocess_logits(
    self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
  ) -> torch.Tensor:
    if sampling_batch_info.has_custom_logits_processor:
      apply_custom_logits_processor(logits, sampling_batch_info)

    return logits

  def forward(self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo) -> torch.Tensor:
    """
    对 logits 进行采样，返回采样的 token ids（批量化实现，无 GPU->CPU 同步）

    Args:
        logits: [num_seqs, vocab_size]
        sampling_batch_info: 每个序列的采样信息

    Returns:
        采样的 token ids: [num_seqs]
    """
    logits = self._preprocess_logits(logits, sampling_batch_info)
    return batched_sample(
      logits,
      sampling_batch_info.temperature,
      sampling_batch_info.top_ps,
      sampling_batch_info.top_ks,
      max_top_k=sampling_batch_info.max_top_k,
    )


def apply_custom_logits_processor(
  logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
) -> torch.Tensor:
  for _, (
    processor,
    batch_mask,
  ) in sampling_batch_info.custom_logit_processor.items():
    # Get the batch indices that need to be processed
    batch_indices = batch_mask.nonzero(as_tuple=True)[0]

    assert batch_mask.shape[0] == len(sampling_batch_info), (
      f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
      f"sampling_batch_info ({len(sampling_batch_info)})"
    )
    batch_mask = torch.repeat_interleave(batch_mask, 1)

    # Apply the processor to the logits
    logits[batch_mask] = processor(
      logits[batch_mask],
      [sampling_batch_info.custom_params[i] for i in batch_indices],
    )

    logger.debug(f"Custom logit processor {processor.__class__.__name__} is applied.")


def greedy_sample(logprobs: torch.Tensor):
  return torch.argmax(logprobs, axis=-1, keepdim=True)


def temperature_sample(logprobs: torch.Tensor, temp: float):
  probs = softmax(logprobs, axis=-1)
  idxs = torch.multinomial(probs, num_samples=1)
  return idxs


def top_k_sample(logprobs: torch.Tensor, k: int):
  topk_logprobs, _ = torch.topk(logprobs, k=k, dim=-1)
  min_topk_logprob = topk_logprobs[..., -1, None]
  logprobs = torch.where(
    logprobs < min_topk_logprob,
    torch.full_like(logprobs, float("-inf")),
    logprobs,
  )
  probs = softmax(logprobs, axis=-1)
  idxs = torch.multinomial(probs, num_samples=1)

  return idxs


def top_p_sample(logprobs: torch.Tensor, p: float):
  # 1. sort the logprobs from largest to smallest
  sorted_logprobs, sorted_indices = torch.sort(logprobs, descending=True, dim=-1)
  # 2. compute cumulative probabilities
  cumulative_probs = torch.cumsum(softmax(sorted_logprobs, axis=-1), dim=-1)
  # 3. truncate tokens with cumulative prob above the threshold
  sorted_indices_to_remove = cumulative_probs > p
  # Keep the first token above the threshold
  if sorted_indices_to_remove[..., 1:].any():
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False  # like [False, False, True(keep last), True, True...]

  indices_to_remove = torch.zeros_like(logprobs, dtype=torch.bool).scatter_(
    dim=-1,
    index=sorted_indices,
    src=sorted_indices_to_remove,
  )

  logprobs = torch.where(
    indices_to_remove,
    torch.full_like(logprobs, float("-inf")),
    logprobs,
  )
  probs = softmax(logprobs, axis=-1)
  idxs = torch.multinomial(probs, num_samples=1)

  return idxs


SAMPLE_IMPLEMENTATIONS: dict[str, Callable] = {
  "greedy": greedy_sample,
  "temperature": temperature_sample,
  "top_k": top_k_sample,
  "top_p": top_p_sample,
}


def make_sampler(temp: float, top_p: float, top_k: int | None):
  def sample(logprobs: torch.Tensor):
    """This way will implement temperature sampling, top-k sampling, and top-p (nucleus) sampling."""
    if temp == 0:
      return greedy_sample(logprobs)
    else:
      logprobs = logprobs / temp
      if top_k is not None and top_k > 0:
        idxs = top_k_sample(logprobs, top_k)
      elif top_p < 1.0:
        idxs = top_p_sample(logprobs, top_p)
      else:
        idxs = temperature_sample(logprobs, temp)
      return idxs

  return sample


# ============================================================================
# Batched Sampling Implementation (Zero GPU->CPU sync)
# ============================================================================


def batched_sample(
  logits: torch.Tensor,
  temperatures: torch.Tensor,
  top_ps: torch.Tensor,
  top_ks: torch.Tensor,
  *,
  max_top_k: int | None = None,
) -> torch.Tensor:
  """
  批量化采样实现，尽可能在 GPU 上执行，避免不必要的 GPU->CPU 同步。

  根据不同采样策略分组处理：
  - greedy (temp == 0)
  - top_k (temp > 0, top_k > 0)
  - top_p (temp > 0, top_k <= 0, top_p < 1.0)
  - temperature (temp > 0, top_k <= 0, top_p >= 1.0)

  Args:
      logits: [batch_size, vocab_size]
      temperatures: [batch_size] 温度参数
      top_ps: [batch_size] top-p 参数
      top_ks: [batch_size] top-k 参数

  Returns:
      next_token_ids: [batch_size]
  """
  batch_size = logits.shape[0]
  device = logits.device
  next_token_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

  # 构建采样策略掩码 (全在 GPU 上计算)
  is_greedy = temperatures == 0
  is_top_k = (~is_greedy) & (top_ks > 0)
  is_top_p = (~is_greedy) & (~is_top_k) & (top_ps < 1.0)
  is_temp_only = (~is_greedy) & (~is_top_k) & (~is_top_p)

  # 1. Greedy sampling (temp == 0)
  # Avoid `if is_greedy.any():` which would trigger `Tensor.__bool__` -> sync on CUDA.
  greedy_logits = logits[is_greedy]
  if greedy_logits.numel() != 0:
    next_token_ids[is_greedy] = torch.argmax(greedy_logits, dim=-1)

  # 2. Top-k sampling
  topk_logits = logits[is_top_k]
  topk_temps = temperatures[is_top_k].unsqueeze(-1)
  topk_ks = top_ks[is_top_k]

  # Apply temperature
  scaled_logits = topk_logits / topk_temps

  # Batched top-k: use a CPU-known max_k to avoid GPU->CPU sync.
  # max_top_k should be passed from ForwardBatch.sampling_max_top_k (pre-computed on CPU).
  if topk_logits.numel() != 0:
    if max_top_k is not None:
      max_k = int(max_top_k)
    else:
      # Avoid any GPU->CPU sync here (e.g., `topk_ks.max().item()`), which would
      # serialize overlap. Callers should pass a CPU-known `max_top_k` computed
      # during batch construction (see ForwardBatch.sampling_max_top_k).
      raise ValueError(
        "batched_sample: `max_top_k` is required for batched top-k sampling to avoid CUDA sync. "
        "Pass `ForwardBatch.sampling_max_top_k` into `SamplingBatchInfo.max_top_k`."
      )
    max_k = max(1, min(max_k, logits.size(-1)))

    topk_vals, topk_idx = torch.topk(scaled_logits, k=max_k, dim=-1)

    # 对每个序列，只保留其 top_k 个值，其他设为 -inf
    k_range = torch.arange(max_k, device=device).unsqueeze(0)  # [1, max_k]
    # Clamp per-seq top_k to a valid range for masking.
    topk_ks_clamped = torch.clamp(topk_ks, min=0, max=max_k).unsqueeze(-1)
    k_mask = k_range < topk_ks_clamped  # [num_topk_seqs, max_k]

    # 将不在 top_k 内的值设为 -inf
    topk_vals = torch.where(k_mask, topk_vals, float("-inf"))

    # Softmax + multinomial
    probs = torch.softmax(topk_vals, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [num_topk_seqs]

    # 从 topk_idx 中取出实际的 token id
    sampled_tokens = topk_idx.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)
    next_token_ids[is_top_k] = sampled_tokens

  # 3. Top-p (nucleus) sampling
  topp_logits = logits[is_top_p]
  topp_temps = temperatures[is_top_p].unsqueeze(-1)
  topp_ps = top_ps[is_top_p].unsqueeze(-1)

  if topp_logits.numel() != 0:
    # Apply temperature
    scaled_logits = topp_logits / topp_temps

    # Sort descending
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)

    # Cumulative probabilities
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens beyond top-p threshold (keep first token above threshold)
    sorted_mask = cumulative_probs > topp_ps
    # Shift right to keep the first token that exceeds threshold
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Set masked tokens to -inf
    sorted_logits = torch.where(sorted_mask, float("-inf"), sorted_logits)

    # Sample from filtered distribution
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Map back to original vocab indices
    sampled_tokens = sorted_indices.gather(dim=-1, index=sampled_idx.unsqueeze(-1)).squeeze(-1)
    next_token_ids[is_top_p] = sampled_tokens

  # 4. Temperature-only sampling (no top-k/top-p filtering)
  temp_logits = logits[is_temp_only]
  temps = temperatures[is_temp_only].unsqueeze(-1)

  if temp_logits.numel() != 0:
    # Apply temperature and sample
    scaled_logits = temp_logits / temps
    probs = torch.softmax(scaled_logits, dim=-1)
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    next_token_ids[is_temp_only] = sampled_tokens

  return next_token_ids
