from .base import PretrainedConfig


class Qwen3Config(PretrainedConfig):
  """
  Qwen3 模型配置类

  与 Qwen2 的主要区别:
  - head_dim 可独立配置（不再由 hidden_size // num_attention_heads 决定）
  - QKV 投影无 bias (attention_bias=False)
  - 引入 QK-Norm（对 Q/K 做 per-head RMSNorm）
  """

  model_type = "qwen3"

  def __init__(
    self,
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    hidden_act="silu",
    max_position_embeddings=40960,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    tie_word_embeddings=True,
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
    self.head_dim = head_dim

    if num_key_value_heads is None:
      num_key_value_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads

    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.attention_bias = attention_bias
    self.attention_dropout = attention_dropout

    super().__init__(
      tie_word_embeddings=tie_word_embeddings,
      use_cache=use_cache,
      dtype=dtype,
      **kwargs,
    )

  @classmethod
  def from_hf_config(cls, hf_config):
    """从 HuggingFace 配置对象创建 Qwen3Config"""
    return cls(
      vocab_size=hf_config.vocab_size,
      hidden_size=hf_config.hidden_size,
      intermediate_size=hf_config.intermediate_size,
      num_hidden_layers=hf_config.num_hidden_layers,
      num_attention_heads=hf_config.num_attention_heads,
      num_key_value_heads=hf_config.num_key_value_heads,
      head_dim=getattr(
        hf_config,
        "head_dim",
        hf_config.hidden_size // hf_config.num_attention_heads,
      ),
      hidden_act=getattr(hf_config, "hidden_act", "silu"),
      max_position_embeddings=hf_config.max_position_embeddings,
      initializer_range=getattr(hf_config, "initializer_range", 0.02),
      rms_norm_eps=hf_config.rms_norm_eps,
      rope_theta=hf_config.rope_theta,
      rope_scaling=getattr(hf_config, "rope_scaling", None),
      attention_bias=getattr(hf_config, "attention_bias", False),
      attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
      tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
      use_cache=getattr(hf_config, "use_cache", True),
      dtype=(str(hf_config.dtype).split(".")[-1] if hasattr(hf_config, "dtype") else "float32"),
    )
