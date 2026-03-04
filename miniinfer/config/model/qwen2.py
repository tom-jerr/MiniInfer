"""
idea and most code from tranformers
"""

import json
from pathlib import Path
from typing import Any
from .base import PretrainedConfig


class Qwen2Config(PretrainedConfig):
  """
  Qwen2 模型配置类

  Args:
      vocab_size: 词汇表大小
      hidden_size: 隐藏层维度
      intermediate_size: MLP 中间层维度
      num_hidden_layers: Transformer 层数
      num_attention_heads: 注意力头数
      num_key_value_heads: KV 注意力头数(用于 GQA)
      hidden_act: 激活函数类型
      max_position_embeddings: 最大位置编码长度
      initializer_range: 权重初始化范围
      rms_norm_eps: RMSNorm 的 epsilon 值
      rope_theta: RoPE 的 theta 参数
      rope_scaling: RoPE 缩放配置
      attention_dropout: 注意力 dropout 概率
      tie_word_embeddings: 是否共享输入输出词嵌入
      use_cache: 是否使用 KV cache
      dtype: 模型精度类型
  """

  model_type = "qwen2"

  def __init__(
    self,
    vocab_size=151936,
    hidden_size=4096,
    intermediate_size=22016,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,
    hidden_act="silu",
    max_position_embeddings=32768,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_dropout=0.0,
    tie_word_embeddings=False,
    use_cache=True,
    dtype="float32",
    **kwargs,
  ):
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads

    # for backward compatibility
    if num_key_value_heads is None:
      num_key_value_heads = num_attention_heads

    self.num_key_value_heads = num_key_value_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.attention_dropout = attention_dropout

    super().__init__(
      tie_word_embeddings=tie_word_embeddings,
      use_cache=use_cache,
      dtype=dtype,
      **kwargs,
    )

  @property
  def head_dim(self) -> int:
    """每个注意力头的维度"""
    return self.hidden_size // self.num_attention_heads

  @classmethod
  def from_hf_config(cls, hf_config):
    """从 HuggingFace 配置对象创建 Qwen2Config"""
    return cls(
      vocab_size=hf_config.vocab_size,
      hidden_size=hf_config.hidden_size,
      intermediate_size=hf_config.intermediate_size,
      num_hidden_layers=hf_config.num_hidden_layers,
      num_attention_heads=hf_config.num_attention_heads,
      num_key_value_heads=hf_config.num_key_value_heads,
      hidden_act=getattr(hf_config, "hidden_act", "silu"),
      max_position_embeddings=hf_config.max_position_embeddings,
      initializer_range=getattr(hf_config, "initializer_range", 0.02),
      rms_norm_eps=hf_config.rms_norm_eps,
      rope_theta=hf_config.rope_theta,
      rope_scaling=getattr(hf_config, "rope_scaling", None),
      attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
      tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
      use_cache=getattr(hf_config, "use_cache", True),
      dtype=(str(hf_config.dtype).split(".")[-1] if hasattr(hf_config, "dtype") else "float32"),
    )


# 预定义的模型配置
class TestQwen2Config:
  """Qwen2 系列模型的预定义配置"""

  @staticmethod
  def qwen2_0_5b() -> Qwen2Config:
    """Qwen2-0.5B 配置"""
    return Qwen2Config(
      vocab_size=151936,
      hidden_size=896,
      intermediate_size=4864,
      num_hidden_layers=24,
      num_attention_heads=14,
      num_key_value_heads=2,
      max_position_embeddings=32768,
      rope_theta=1000000.0,
      rms_norm_eps=1e-6,
      tie_word_embeddings=True,
    )

  @staticmethod
  def qwen2_1_5b() -> Qwen2Config:
    """Qwen2-1.5B 配置"""
    return Qwen2Config(
      vocab_size=151936,
      hidden_size=1536,
      intermediate_size=8960,
      num_hidden_layers=28,
      num_attention_heads=12,
      num_key_value_heads=2,
      max_position_embeddings=32768,
      rope_theta=1000000.0,
      rms_norm_eps=1e-6,
      tie_word_embeddings=True,
    )

  @staticmethod
  def qwen2_7b() -> Qwen2Config:
    """Qwen2-7B 配置"""
    return Qwen2Config(
      vocab_size=152064,
      hidden_size=3584,
      intermediate_size=18944,
      num_hidden_layers=28,
      num_attention_heads=28,
      num_key_value_heads=4,
      max_position_embeddings=32768,
      rope_theta=1000000.0,
      rms_norm_eps=1e-6,
      tie_word_embeddings=False,
    )


__all__ = ["PretrainedConfig", "Qwen2Config", "TestQwen2Config"]
