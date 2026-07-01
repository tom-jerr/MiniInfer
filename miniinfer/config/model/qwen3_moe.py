"""Qwen3-MoE 模型配置类。

Qwen3-MoE 在 Qwen3 dense 基础上把（部分）层的 MLP 替换为 MoE：
- attention 路径与 Qwen3 dense 一致（qkv 融合、QK-Norm、attention_bias=False）；
- MLP 路径：按 ``decoder_sparse_step`` / ``mlp_only_layers`` / ``first_k_dense_replace``
  决定该层是 MoE 还是 dense MLP。

HF config 关键字段（Qwen3MoeConfig）：
- ``num_experts``：路由专家总数；
- ``num_experts_per_tok``：每个 token 选中的专家数 top_k；
- ``moe_intermediate_size``：单个专家的 intermediate_size；
- ``decoder_sparse_step``：每隔多少层用一个 MoE（默认 1 = 每层都是 MoE）；
- ``mlp_only_layers``：显式指定用 dense MLP 的层索引列表；
- ``first_k_dense_replace``：前 K 层强制用 dense MLP（与 decoder_sparse_step 取并集）；
- ``norm_topk_prob``：是否对 topk 概率归一化（Qwen3-MoE 默认 True）；
- ``routed_scaling_factor``：路由缩放因子（保留字段，当前 forward 暂未单独使用）。
"""
from .base import PretrainedConfig


class Qwen3MoeConfig(PretrainedConfig):
  model_type = "qwen3_moe"

  def __init__(
    self,
    vocab_size=151936,
    hidden_size=2048,
    intermediate_size=6144,  # dense MLP 的 intermediate_size（mlp_only_layers 用）
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
    # ---- MoE 专属 ----
    num_experts=128,
    num_experts_per_tok=8,
    moe_intermediate_size=768,
    decoder_sparse_step=1,
    mlp_only_layers=None,
    first_k_dense_replace=0,
    norm_topk_prob=True,
    routed_scaling_factor=1.0,
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

    # MoE 专属
    self.num_experts = num_experts
    self.num_experts_per_tok = num_experts_per_tok
    self.moe_intermediate_size = moe_intermediate_size
    self.decoder_sparse_step = decoder_sparse_step
    self.mlp_only_layers = list(mlp_only_layers) if mlp_only_layers else []
    self.first_k_dense_replace = first_k_dense_replace
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor

    super().__init__(
      tie_word_embeddings=tie_word_embeddings,
      use_cache=use_cache,
      dtype=dtype,
      **kwargs,
    )

  @classmethod
  def from_hf_config(cls, hf_config):
    """从 HuggingFace 配置对象创建 Qwen3MoeConfig。"""
    return cls(
      vocab_size=hf_config.vocab_size,
      hidden_size=hf_config.hidden_size,
      intermediate_size=getattr(hf_config, "intermediate_size", 0),
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
      rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
      rope_scaling=getattr(hf_config, "rope_scaling", None),
      attention_bias=getattr(hf_config, "attention_bias", False),
      attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
      tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", True),
      use_cache=getattr(hf_config, "use_cache", True),
      dtype=(str(hf_config.dtype).split(".")[-1] if hasattr(hf_config, "dtype") else "float32"),
      num_experts=getattr(hf_config, "num_experts", 0) or 0,
      num_experts_per_tok=getattr(hf_config, "num_experts_per_tok", 1) or 1,
      moe_intermediate_size=getattr(hf_config, "moe_intermediate_size", 0) or 0,
      decoder_sparse_step=getattr(hf_config, "decoder_sparse_step", 1) or 1,
      mlp_only_layers=getattr(hf_config, "mlp_only_layers", None) or [],
      # HF 可能为 None（表示不强制前 K 层 dense）；统一成 0。
      first_k_dense_replace=getattr(hf_config, "first_k_dense_replace", 0) or 0,
      norm_topk_prob=getattr(hf_config, "norm_topk_prob", True),
      routed_scaling_factor=getattr(hf_config, "routed_scaling_factor", 1.0) or 1.0,
    )
