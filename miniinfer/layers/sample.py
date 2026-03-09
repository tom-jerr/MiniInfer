from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch

from .linear import softmax

logger = logging.getLogger(__name__)


@dataclass
class BatchSamplingArgs:
  """
  mini-sglang-style sampling args:
  - temperatures is None: the whole batch is greedy (fast-path argmax)
  - top_k/top_p are optional: omitted when disabled for the whole batch

  Note: when using flashinfer, we will clamp temperature/top_p and map top_k<=0
  to vocab_size on-device (in-place) to avoid per-step CPU work.
  """

  temperatures: torch.Tensor | None
  top_k: torch.Tensor | int | None = None
  top_p: torch.Tensor | float | None = None
  # CPU-side maximum top-k used to avoid GPU->CPU sync in the torch fallback.
  max_top_k: Optional[int] = None


def _is_sm90_supported(device: torch.device) -> bool:
  if device.type != "cuda" or not torch.cuda.is_available():
    return False
  try:
    major, minor = torch.cuda.get_device_capability(device)
  except Exception:
    return False
  return major == 9 and minor == 0


class Sampler:
  """
  Sampler redesigned after mini-sglang:
  - metadata is prepared/transported by ForwardBatch (pinned H2D on schedule stream)
  - sampling chooses flashinfer kernels when available, otherwise falls back to torch
  """

  def __init__(
    self,
    *,
    device: Union[str, torch.device] = "cuda",
    vocab_size: int = 0,
  ):
    self.device = torch.device(device)
    self.vocab_size = int(vocab_size)
    

  def prepare(self, *, temperatures: torch.Tensor | None, top_k, top_p, max_top_k: int | None):
    return BatchSamplingArgs(
      temperatures=temperatures,
      top_k=top_k,
      top_p=top_p,
      max_top_k=max_top_k,
    )

  def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor: 
    import flashinfer.sampling as sampling
    probs = sampling.softmax(logits, args.temperatures, enable_pdl=_is_sm90_supported())
    if args.top_k is None and args.top_p is None:
        return sampling.sampling_from_probs(probs)

    if args.top_p is None:
        assert args.top_k is not None
        return sampling.top_k_sampling_from_probs(probs, args.top_k)

    if args.top_k is None:
        assert args.top_p is not None
        return sampling.top_p_sampling_from_probs(probs, args.top_p)

    assert args.top_k is not None and args.top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, args.top_k, args.top_p)
