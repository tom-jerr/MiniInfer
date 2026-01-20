from typing import Any, Union

import torch
import torch.nn as nn


def dequantize_linear(torch_layer: Union[nn.Linear, nn.Embedding, Any]) -> torch.Tensor:
  """
  反量化线性层权重,适配Hugging Face Transformers模型

  对于Qwen2-0.5B-Instruct这种标准模型, 通常权重已经是未量化的，
  但需要处理可能的量化场景（如用户手动量化后的模型）

  Args:
      torch_layer: PyTorch线性层或Embedding层

  Returns:
      反量化后的权重tensor (未量化则直接返回原权重)
  """

  # 情况1: 标准的 nn.Linear 或 nn.Embedding (未量化)
  if isinstance(torch_layer, (nn.Linear, nn.Embedding)):
    return torch_layer.weight.data.clone()

  # 情况2: torch.ao.quantization 量化的层
  if hasattr(torch_layer, "weight") and callable(torch_layer.weight):
    try:
      # torch.quantized.Linear 的 weight() 返回量化tensor
      weight = torch_layer.weight()
      if hasattr(weight, "dequantize"):
        return weight.dequantize()
      return weight
    except:
      pass
  # 尝试直接访问weight属性
  if hasattr(torch_layer, "weight"):
    return torch_layer.weight.data.clone()
  raise ValueError(
    f"Cannot dequantize layer of type {type(torch_layer)}. " f"Layer attributes: {dir(torch_layer)}"
  )
