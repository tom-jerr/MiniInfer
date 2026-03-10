import torch
from torch import nn
from typing import Dict, Optional

from kernels.triton.rms_norm import add_rms_norm_forward
from miniinfer.loader.weight import WeightLoaderMixin


class RMSNorm(nn.Module, WeightLoaderMixin):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = None

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    weight_key = f"{prefix}.weight"
    assert weight_key in state_dict, f"Weight key {weight_key} not found in state_dict."
    self.weight = self._to_device(state_dict[weight_key], device, dtype)
    assert self.weight is not None, "Weight must be loaded before forward pass."

  @torch.compile
  def rms_forward(
    self,
    x: torch.Tensor,
  ) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.eps)
    # Match HF implementation: convert back to input_dtype before multiplying weight
    return self.weight * x.to(input_dtype)

  def forward(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if residual is not None:
      out, residual_out = add_rms_norm_forward(x, residual, self.weight, self.eps)
      return out, residual_out
    return self.rms_forward(x)
