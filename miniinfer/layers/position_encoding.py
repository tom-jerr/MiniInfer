import torch
import torch.nn as nn
from kernels.triton.rotary_embedding import apply_rotary_embedding
from typing import Union, Optional, Tuple


class RotaryEmbedding(nn.Module):
  def __init__(
    self,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool = True,
    dtype: torch.dtype = torch.float16,
  ):
    super().__init__()
    self.head_size = head_size
    self.rotary_dim = rotary_dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    self.is_neox_style = is_neox_style
    self.dtype = dtype
    cache = self._compute_cos_sin_cache()
    self.cos_sin_cache: torch.Tensor
    self.register_buffer("cos_sin_cache", cache, persistent=False)
    self._apply_rotary_emb_wrapped = _apply_rotary_emb

  def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
    inv_freq = 1.0 / (
      base
      ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cpu") / self.rotary_dim)
    )
    return inv_freq

  def _compute_cos_sin_cache(self) -> torch.Tensor:
    """Compute the cos and sin cache."""
    inv_freq = self._compute_inv_freq(self.base)
    t = torch.arange(self.max_position_embeddings, dtype=torch.float, device="cpu")

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

  def forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key.
    Args:
        positions: [num_tokens], flatten batch and seq_len
        query: [num_tokens, num_heads, head_size]
        key: [num_tokens, num_heads, head_size]
        offsets: Optional tensor to offset positions (for caching)
    """
    if offsets is not None:
      positions = positions + offsets
    positions = positions.flatten()
    num_tokens = positions.shape[0]
    # Ensure positions is on the same device as cos_sin_cache
    positions_device = positions.device
    if self.cos_sin_cache.device != positions_device:
      # Move positions to cache's device for indexing, then move result back
      positions_cpu = positions.cpu()
      cos_sin = self.cos_sin_cache.index_select(0, positions_cpu).to(positions_device)
    else:
      cos_sin = self.cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    # maybe some model just apply rotary to part of the head dim
    query_rot = query[..., : self.rotary_dim]
    query_pass = query[..., self.rotary_dim :]
    query_rot = self._apply_rotary_emb_wrapped(query_rot, cos, sin, self.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)
    key_rot = key[..., : self.rotary_dim]
    key_pass = key[..., self.rotary_dim :]
    key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def _apply_rotary_emb(
  x: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  is_neox_style: bool,
) -> torch.Tensor:
  """
  Args:
      x: [num_tokens, num_heads, head_size]
      cos: [num_tokens, head_size // 2]
      sin: [num_tokens, head_size // 2]
      is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
          positional embeddings.
  """
  cos = cos.unsqueeze(-2).to(x.dtype)
  sin = sin.unsqueeze(-2).to(x.dtype)
  if is_neox_style:
    x1, x2 = torch.chunk(x, 2, dim=-1)
  else:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
  o1 = x1 * cos - x2 * sin
  o2 = x2 * cos + x1 * sin
  if is_neox_style:
    return torch.cat((o1, o2), dim=-1)
  else:
    return torch.stack((o1, o2), dim=-1).flatten(-2)
