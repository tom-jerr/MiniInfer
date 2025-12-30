from abc import ABC, abstractmethod

import torch

from src import causal_mask


class TinyKvCache(ABC):
  @abstractmethod
  def update_and_fetch(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_length: int | None = None,
    mask: torch.Tensor | str | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    """
    Update the key-value cache and fetch the updated key-value cache.

    Args:
        key: The key to update the cache with.
        value: The value to update the cache with.
        mask_length: The length of the mask (only used in batching mode)
        mask: The mask to use (only used in batching mode)

    Returns:
        A tuple of the updated key cache, the updated value cache, the sequence length, and the mask.
    """


class BatchingKvCache(TinyKvCache):
  """A KV cache that supports batching multiple requests with different sequence lengths.
  Just padding the keys and values to the `max_seq_len` in the batch.
  """

  def __init__(self, max_active_requests: int, max_seq_len: int):
    self.max_active_requests = max_active_requests
    self.max_seq_len = max_seq_len
    self.kv_caches: list[TinyKvCache] = [None] * max_active_requests
    self.HD = None

  def update_and_fetch(
    self,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask_length: int | None = None,
    mask: torch.Tensor | str | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
    B, H, S, D = keys.shape
    assert keys.shape == values.shape
    assert self.max_seq_len >= S
    if self.HD is None:
      self.HD = (H, D)
    else:
      assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"

    assert self.max_active_requests == B
    # 1. update cache
    max_seq_len = 0
    data = [None] * B  # to store per-request data
    for b in range(B):
      if self.kv_caches[b] is None:
        continue
      key, value = keys[b : b + 1], values[b : b + 1]
      new_key, new_value, seq_len, old_mask = self.kv_caches[b].update_and_fetch(key, value)
      data[b] = (new_key[0], new_value[0], seq_len, old_mask)
      max_seq_len = max(max_seq_len, seq_len)

    # 2. allocate batched buffers (use input tensors' dtype/device)
    batched_keys = torch.zeros(
      (self.max_active_requests, H, max_seq_len, D),
      dtype=keys.dtype,
      device=keys.device,
    )
    batched_values = torch.zeros(
      (self.max_active_requests, H, max_seq_len, D),
      dtype=values.dtype,
      device=values.device,
    )

    if mask_length is None:
      masks = None
    else:
      masks = torch.full(
        (self.max_active_requests, mask_length, max_seq_len),
        -torch.inf,
        dtype=keys.dtype,
        device=keys.device,
      )

    # 3. Fill per-request data aligned to the end and set masks
    for i, entry in enumerate(data):
      if entry is None:
        continue
      k_i, v_i, s_i, _ = entry
      # 尾部对齐
      batched_keys[i, :, max_seq_len - s_i : max_seq_len, :] = k_i
      batched_values[i, :, max_seq_len - s_i : max_seq_len, :] = v_i
      if masks is not None:
        if mask == "causal" or mask is None:
          masks[i, :, max_seq_len - s_i : max_seq_len] = causal_mask(
            mask_length, s_i, dtype=keys.dtype, device=keys.device
          )
        elif isinstance(mask, torch.Tensor):
          masks[i, :, max_seq_len - s_i : max_seq_len] = mask
        else:
          raise NotImplementedError(f"Unsupported mask type: {type(mask)}")
    return (
      batched_keys,
      batched_values,
      max_seq_len,
      masks.reshape(B, 1, mask_length, max_seq_len),
    )

  def add_request(self, prefilled: TinyKvCache, id: int):
    if id >= self.max_active_requests:
      return ValueError(f"Request id {id} out of range")
    if getattr(prefilled, "key_values", None) is not None:
      keys, _ = prefilled.key_values
      B, H, _, D = keys.shape
      assert B == 1
      if self.HD is None:
        self.HD = (H, D)
      else:
        assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
    self.kv_caches[id] = prefilled

  def remove_request(self, id: int):
    if self.kv_caches is None:
      raise ValueError(f"Request id {id} is not in the cache")
    self.kv_caches[id] = None


class TinyKvFullCache:
  """The simplest KV cache that keeps all keys and values in memory"""

  def __init__(self):
    self.key_values = None
    self.offset = 0

  @torch.no_grad()
  def update_and_fetch(
    self,
    key: torch.Tensor,
    value: torch.Tensor,
    mask_length: int | None = None,
    mask: torch.Tensor | str | None = None,
  ):
    key = key.detach()  # do not keep autograd
    value = value.detach()
    B, H, S, D = key.shape

    if self.key_values is None:
      self.key_values = (key, value)
      self.offset = S
    else:
      prev_key, prev_value = self.key_values
      new_key = torch.cat([prev_key, key], dim=2)
      new_value = torch.cat([prev_value, value], dim=2)
      self.key_values = (new_key, new_value)
      self.offset += S

    return self.key_values[0], self.key_values[1], self.offset, mask
