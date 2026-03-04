from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class BaseModelOutput:
  """
  模型输出类。

  Args:
      logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
          分类层的预测分值（映射回词表空间后的结果）。

      last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
          最后一层 Transformer Block 输出的隐藏状态。

      hidden_states (`Optional[Tuple[torch.FloatTensor, ...]]`):
          每一层输出的隐藏状态（包含 Embedding 输出, do not contain the last layer hidden_states）。
  """

  logits: torch.FloatTensor = None
  last_hidden_state: Optional[torch.FloatTensor] = None
  hidden_states: Optional[List[torch.FloatTensor]] = None
