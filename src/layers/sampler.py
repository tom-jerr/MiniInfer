import torch
from typing import Callable
from .linear import softmax


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
