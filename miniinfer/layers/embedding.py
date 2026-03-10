import torch
from torch import nn
from typing import Dict

from miniinfer.loader.weight import WeightLoaderMixin

from .linear import linear


# TODO(lzy): tp and pp support
class VocabEmbedding(nn.Module, WeightLoaderMixin):
  def __init__(self, vocab_size: int, embedding_dim: int):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    # 初始化为 None，等待 load_weights 填充
    self.weight = None

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    weight_key = f"{prefix}.weight"
    self.weight = self._to_device(state_dict[weight_key], device, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # embedding lookup
    # self.weight: [vocab_size, embedding_dim]
    assert self.weight is not None, "Weight must be loaded before forward pass."
    return self.weight[x, :]

  def as_linear(self, x: torch.Tensor) -> torch.Tensor:
    # for LM Head usage
    assert self.weight is not None, "Weight must be loaded before as_linear pass."
    return linear(x, self.weight)


class LMHead(VocabEmbedding):
  """Language Model Head，继承自 VocabEmbedding，用于将隐藏状态映射回词汇表空间。"""

  def __init__(self, vocab_size: int, embedding_dim: int):
    super().__init__(vocab_size, embedding_dim)

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    # 优先尝试加载 prefix 指定的权重（例如 'lm_head'）
    try:
      super().load_weights(state_dict, prefix, device, dtype)
    except KeyError:
      has_model_prefix = "model.embed_tokens.weight" in state_dict
      p = "model." if has_model_prefix else ""
      super().load_weights(state_dict, f"{p}embed_tokens", device, dtype)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 重用 VocabEmbedding 的 as_linear 方法
    return self.as_linear(x)
