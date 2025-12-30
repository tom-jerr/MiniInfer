from typing import Any, Optional

import torch
import torch.nn as nn

from ..engine.kv_cache import TinyKvCache, TinyKvFullCache
from ..layers import (
  LMHead,
  RMSNorm,
  RotaryEmbedding,
  get_activation,
  get_attention,
  linear,
)
from ..utils.quantize import dequantize_linear
from ..config.model.qwen2 import Qwen2Config


class Qwen2Attention(nn.Module):
  def __init__(
    self,
    config: Qwen2Config,
  ):
    super().__init__()
    self.config = config

    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(
      config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
    )
    self.k_proj = nn.Linear(
      config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.v_proj = nn.Linear(
      config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.o_proj = nn.Linear(
      config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
    )

    self.rope = RotaryEmbedding(
      self.head_dim,
      config.max_position_embeddings,
      config.rope_theta,
      traditional=False,
    )  # Qwen2 uses nontraditional RoPE

  def forward(
    self,
    hidden_states: torch.Tensor,
    past_key_value: Optional[TinyKvCache] = None,
    mask: torch.Tensor | str | None = None,
    use_cache: bool = False,
    offset: list[slice] | slice | None = None,
  ) -> tuple[torch.Tensor, Optional[TinyKvCache]]:
    """
    Args:
        hidden_states: (B, L, E)
        past_key_value: cache for this layer
        mask: (B, 1, L, S) or "causal" or None
        use_cache: whether to return updated cache
        offset: manual position offset (for backward compatibility)

    Returns:
        tuple of (attn_output, updated_cache)
    """
    input_shape = hidden_states.shape[:-1]
    # [B, S, H] → [B, S, num_heads, head_dim]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # calc q, k, v
    query_states = self.q_proj(hidden_states).view(hidden_shape)
    key_states = self.k_proj(hidden_states).view(hidden_shape)
    value_states = self.v_proj(hidden_states).view(hidden_shape)

    # rope
    query_states = self.rope(
      query_states,
      offset=offset,
    )
    key_states = self.rope(
      key_states,
      offset=offset,
    )

    # [B, S, num_heads, head_dim] -> [B, num_heads, S, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # Update cache if provided
    if past_key_value is not None:
      key_states, value_states, _, mask = past_key_value.update_and_fetch(
        key_states, value_states, mask_length=input_shape[1], mask=mask
      )

    attn_output = get_attention("gqa")(
      query_states,
      key_states,
      value_states,
      scale=self.scaling,
      mask=mask,
    )
    # attn_output shape: [B, num_heads, S, head_dim]
    # need to transpose to [B, S, num_heads, head_dim] then reshape to [B, S, hidden_size]
    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, past_key_value if use_cache else None


class Qwen2MLP(nn.Module):
  def __init__(
    self,
    config: Qwen2Config,
  ):
    super().__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = get_activation(config.hidden_act)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """MLP(x)=(SiLU(W_gate(x))⊙W_up(x))W_down"""
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


class Qwen2TransformerBlock(nn.Module):
  def __init__(
    self,
    config: Qwen2Config,
    w_input_layernorm: torch.Tensor,
    w_post_attention_layernorm: torch.Tensor,
  ):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = Qwen2Attention(config=config)
    self.mlp = Qwen2MLP(config)
    self.input_layernorm = RMSNorm(
      config.hidden_size, weight=w_input_layernorm, eps=config.rms_norm_eps
    )
    self.post_attention_layernorm = RMSNorm(
      config.hidden_size,
      weight=w_post_attention_layernorm,
      eps=config.rms_norm_eps,
    )

  def forward(
    self,
    x: torch.Tensor,
    past_key_value: Optional[TinyKvCache] = None,
    mask: torch.Tensor | str | None = None,
    use_cache: bool = False,
    offset: Optional[int] = None,
  ) -> tuple[torch.Tensor, Optional[TinyKvCache]]:
    attn_output, updated_cache = self.self_attn(
      self.input_layernorm(x),
      past_key_value=past_key_value,
      mask=mask,
      use_cache=use_cache,
      offset=offset,
    )
    h = x + attn_output
    r = self.mlp(self.post_attention_layernorm(h))
    out = h + r
    return out, updated_cache


class Qwen2Model(nn.Module):
  def __init__(
    self,
    torch_model: Any,  # from GPU
  ):
    super().__init__()
    # layer constructed in CPU, so we need to transfer device to GPU here
    precision = torch.float16
    self.precision = precision

    # 获取设备信息
    device = next(torch_model.parameters()).device

    # 从 transformers 模型创建用户配置
    user_config = Qwen2Config(
      vocab_size=torch_model.config.vocab_size,
      hidden_size=torch_model.config.hidden_size,
      intermediate_size=torch_model.config.intermediate_size,
      num_hidden_layers=torch_model.config.num_hidden_layers,
      num_attention_heads=torch_model.config.num_attention_heads,
      num_key_value_heads=torch_model.config.num_key_value_heads,
      hidden_act=torch_model.config.hidden_act,
      max_position_embeddings=torch_model.config.max_position_embeddings,
      rms_norm_eps=torch_model.config.rms_norm_eps,
      rope_theta=torch_model.config.rope_theta,
      attention_dropout=torch_model.config.attention_dropout,
      tie_word_embeddings=torch_model.config.tie_word_embeddings,
    )
    self.config = user_config

    self.embedding = LMHead(
      vocab_size=user_config.vocab_size,
      embedding_dim=user_config.hidden_size,
      weight=dequantize_linear(torch_model.model.embed_tokens).to(precision),
    ).to(device)

    self.layers_inner = nn.ModuleList()

    for i in range(user_config.num_hidden_layers):
      w_input_layernorm = torch_model.model.layers[i].input_layernorm.weight.to(precision)
      w_post_attention_layernorm = torch_model.model.layers[i].post_attention_layernorm.weight.to(
        precision
      )

      layer = (
        Qwen2TransformerBlock(
          config=user_config,
          w_input_layernorm=w_input_layernorm,
          w_post_attention_layernorm=w_post_attention_layernorm,
        )
        .to(device)
        .to(precision)
      )

      # 复制权重到新创建的层
      with torch.no_grad():
        layer.self_attn.q_proj.weight.copy_(torch_model.model.layers[i].self_attn.q_proj.weight)
        layer.self_attn.k_proj.weight.copy_(torch_model.model.layers[i].self_attn.k_proj.weight)
        layer.self_attn.v_proj.weight.copy_(torch_model.model.layers[i].self_attn.v_proj.weight)
        layer.self_attn.o_proj.weight.copy_(torch_model.model.layers[i].self_attn.o_proj.weight)
        layer.self_attn.q_proj.bias.copy_(torch_model.model.layers[i].self_attn.q_proj.bias)
        layer.self_attn.k_proj.bias.copy_(torch_model.model.layers[i].self_attn.k_proj.bias)
        layer.self_attn.v_proj.bias.copy_(torch_model.model.layers[i].self_attn.v_proj.bias)
        layer.mlp.gate_proj.weight.copy_(torch_model.model.layers[i].mlp.gate_proj.weight)
        layer.mlp.up_proj.weight.copy_(torch_model.model.layers[i].mlp.up_proj.weight)
        layer.mlp.down_proj.weight.copy_(torch_model.model.layers[i].mlp.down_proj.weight)

      self.layers_inner.append(layer)

    self.norm = RMSNorm(
      dim=user_config.hidden_size,
      weight=torch_model.model.norm.weight.to(precision),
      eps=user_config.rms_norm_eps,
    ).to(device)

    if not user_config.tie_word_embeddings:
      # 7b 模型有单独的 lm_head 线性层，如果设置使用它来进行最后的映射，否则用 embedding 的 linear 映射回去
      self.register_buffer("w_lm_head", dequantize_linear(torch_model.lm_head).to(precision))
    else:
      self.w_lm_head = None

    self.torch_model = torch_model

  def forward(
    self,
    inputs: torch.Tensor,
    offset: Optional[int] = None,
    cache: Optional[list[TinyKvCache]] = None,  # 旧接口：位置参数
    past_key_values: Optional[list[TinyKvCache]] = None,  # 新接口：关键字参数
    use_cache: bool = False,
    mask: torch.Tensor | str | None = "causal",
  ) -> tuple[torch.Tensor, Optional[list[TinyKvCache]]]:
    """
    Args:
        inputs: input token ids (B, L)
        offset: position offset for RoPE (for backward compatibility)
        cache: list of caches (old interface, positional arg)
        past_key_values: list of caches for each layer (new interface, keyword arg)
        use_cache: whether to return updated caches
        mask: attention mask

    Returns:
        tuple of (logits, updated_caches)

    Calling conventions:
        Old: model(inputs, offset, cache, use_cache=True)
        New: model(inputs, past_key_values=cache, use_cache=True)
    """
    h = self.embedding(inputs)

    # Handle backward compatibility: cache (positional) takes precedence over past_key_values (keyword)
    if cache is not None:
      past_key_values = cache

    # Initialize caches if needed
    if use_cache and past_key_values is None:
      past_key_values = [TinyKvFullCache() for _ in range(len(self.layers_inner))]

    updated_caches = [] if use_cache else None

    for idx, layer in enumerate(self.layers_inner):
      past_kv = past_key_values[idx] if past_key_values is not None else None
      h, updated_cache = layer(
        h, past_key_value=past_kv, mask=mask, use_cache=use_cache, offset=offset
      )
      if use_cache:
        updated_caches.append(updated_cache)

    h = self.norm(h)
    if self.w_lm_head is not None:
      logits = linear(h, self.w_lm_head)
    else:
      logits = self.embedding.as_linear(h)

    return logits, updated_caches if use_cache else None


# from ..cache.kv_cache import TinyKvCache


# class Qwen2MultiHeadAttentionWithCache(nn.Module):
#     def __init__(
#         self,
#         hidden_size: int,
#         num_heads: int,
#         num_kv_heads: int,
#         wq: torch.Tensor,
#         wk: torch.Tensor,
#         wv: torch.Tensor,
#         wo: torch.Tensor,
#         bq: torch.Tensor,
#         bk: torch.Tensor,
#         bv: torch.Tensor,
#         max_seq_len: int = 32768,
#         theta: int = 10000,
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.num_kv_heads = num_kv_heads
#         assert (
#             hidden_size % num_heads == 0
#         ), f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
#         assert (
#             num_heads % num_kv_heads == 0
#         ), f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
#         self.head_dim = hidden_size // num_heads
#         self.scale = 1.0 / (self.head_dim**0.5)
#         # Trainable parameters
#         self.register_buffer("wq", wq)
#         self.register_buffer("wk", wk)
#         self.register_buffer("wv", wv)
#         self.register_buffer("wo", wo)
#         self.register_buffer("bq", bq)
#         self.register_buffer("bk", bk)
#         self.register_buffer("bv", bv)

#         self.rope = RotaryEmbedding(
#             self.head_dim, max_seq_len, theta, traditional=False
#         )  # Qwen2 uses nontraditional RoPE

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask: Optional[torch.Tensor | str] = None,
#         offset: int | None = None,
#         cache: TinyKvCache | None = None,
#     ) -> torch.Tensor:
#         """
#         Args:
#             x: (B, L, E)
#             mask: (B, 1, L, S) or "causal" or None
#             offset: for kv cache
#             cache: for kv cache
#         """
#         B, L, _ = x.shape

#         projection_q = linear(x, self.wq, bias=self.bq).reshape(
#             B, L, self.num_heads, self.head_dim
#         )
#         projection_k = linear(x, self.wk, bias=self.bk).reshape(
#             B, L, self.num_kv_heads, self.head_dim
#         )
#         projection_v = linear(x, self.wv, bias=self.bv).reshape(
#             B, L, self.num_kv_heads, self.head_dim
#         )

#         if isinstance(offset, int):
#             offset_slice = [slice(offset, offset + L)]
#         else:
#             offset_slice = [slice(int(i), int(i + L)) for i in offset]

#         # 只计算新生成的 token
#         projection_q = self.rope(projection_q, offset=offset_slice)
#         projection_k = self.rope(projection_k, offset=offset_slice)

#         projection_q = projection_q.transpose(1, 2)  # (B, num_heads, L, head_dim)
#         projection_k = projection_k.transpose(1, 2)  # (B, num_kv_heads, L, head_dim)
#         projection_v = projection_v.transpose(1, 2)  # (B, num_kv_heads, L, head_dim)

#         # batch kv cache update and fetch
#         projection_k, projection_v, _, mask = cache.update_and_fetch(
#             projection_k, projection_v, mask_length=L, mask=mask
#         )
#         # S = projection_k.shape[-2]

#         x = scaled_dot_product_attention_grouped(
#             projection_q.to(torch.float32),
#             projection_k.to(torch.float32),
#             projection_v.to(torch.float32),
#             scale=self.scale,
#             mask=mask,
#         ).to(x.dtype)

#         x = x.transpose(1, 2).reshape(B, L, self.hidden_size)
#         return linear(x, self.wo)


# class Qwen2TransformerBlockWithCache(nn.Module):
#     def __init__(
#         self,
#         num_attention_heads: int,
#         num_kv_heads: int,
#         hidden_size: int,
#         intermediate_size: int,
#         rms_norm_eps: float,
#         wq: torch.Tensor,
#         wk: torch.Tensor,
#         wv: torch.Tensor,
#         wo: torch.Tensor,
#         bq: torch.Tensor,
#         bk: torch.Tensor,
#         bv: torch.Tensor,
#         w_gate: torch.Tensor,
#         w_up: torch.Tensor,
#         w_down: torch.Tensor,
#         w_input_layernorm: torch.Tensor,
#         w_post_attention_layernorm: torch.Tensor,
#         max_seq_len: int = 32768,
#         theta: int = 10000,
#     ):
#         super().__init__()
#         self.num_attention_heads = num_attention_heads
#         self.hidden_size = hidden_size

#         self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
#         self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
#         self.post_attention_layernorm = RMSNorm(
#             hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
#         )
#         self.self_attn = Qwen2MultiHeadAttentionWithCache(
#             num_heads=num_attention_heads,
#             hidden_size=hidden_size,
#             num_kv_heads=num_kv_heads,
#             wq=wq,
#             wk=wk,
#             wv=wv,
#             wo=wo,
#             bq=bq,
#             bk=bk,
#             bv=bv,
#             max_seq_len=max_seq_len,
#             theta=theta,
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         mask: Optional[torch.Tensor | str] = None,
#         offset: int | None = None,
#         cache: TinyKvCache | None = None,
#     ) -> torch.Tensor:
#         r = self.self_attn(self.input_layernorm(x), mask, offset, cache)
#         h = x + r
#         r = self.mlp(self.post_attention_layernorm(h))
#         out = h + r
#         return out


# class Qwen2ModelV2(nn.Module):
#     def __init__(
#         self,
#         torch_model: Any,
#     ):
#         super().__init__()
#         self.num_hidden_layers = torch_model.config.num_hidden_layers
#         self.hidden_size = torch_model.config.hidden_size
#         self.vocab_size = torch_model.config.vocab_size
#         precision = torch.float16
#         self.precision = precision

#         self.embedding = EmbeddingLayer(
#             vocab_size=self.vocab_size,
#             embedding_dim=self.hidden_size,
#             weight=dequantize_linear(torch_model.model.embed_tokens).to(precision),
#         )

#         self.layers_inner = nn.ModuleList()

#         for i in range(torch_model.config.num_hidden_layers):
#             wq = dequantize_linear(torch_model.model.layers[i].self_attn.q_proj)
#             wk = dequantize_linear(torch_model.model.layers[i].self_attn.k_proj)
#             wv = dequantize_linear(torch_model.model.layers[i].self_attn.v_proj)
#             wo = dequantize_linear(torch_model.model.layers[i].self_attn.o_proj)
#             w_gate = dequantize_linear(torch_model.model.layers[i].mlp.gate_proj)
#             w_up = dequantize_linear(torch_model.model.layers[i].mlp.up_proj)
#             w_down = dequantize_linear(torch_model.model.layers[i].mlp.down_proj)

#             layer = Qwen2TransformerBlockWithCache(
#                 num_attention_heads=torch_model.config.num_attention_heads,
#                 num_kv_heads=torch_model.config.num_key_value_heads,
#                 hidden_size=torch_model.config.hidden_size,
#                 intermediate_size=torch_model.config.intermediate_size,
#                 rms_norm_eps=torch_model.config.rms_norm_eps,
#                 wq=wq.to(precision),
#                 wk=wk.to(precision),
#                 wv=wv.to(precision),
#                 wo=wo.to(precision),
#                 bq=torch_model.model.layers[i].self_attn.q_proj.bias.to(precision),
#                 bk=torch_model.model.layers[i].self_attn.k_proj.bias.to(precision),
#                 bv=torch_model.model.layers[i].self_attn.v_proj.bias.to(precision),
#                 w_gate=w_gate.to(precision),
#                 w_up=w_up.to(precision),
#                 w_down=w_down.to(precision),
#                 w_input_layernorm=torch_model.model.layers[i].input_layernorm.weight.to(
#                     precision
#                 ),
#                 w_post_attention_layernorm=torch_model.model.layers[
#                     i
#                 ].post_attention_layernorm.weight.to(precision),
#                 max_seq_len=torch_model.config.max_position_embeddings,
#                 theta=torch_model.config.rope_theta,
#             )
#             self.layers_inner.append(layer)

#         self.norm = RMSNorm(
#             torch_model.config.hidden_size,
#             weight=torch_model.model.norm.weight.to(precision),
#             eps=torch_model.config.rms_norm_eps,
#         )

#         if not torch_model.config.tie_word_embeddings:
#             # 7b 模型有单独的 lm_head 线性层，如果设置使用它来进行最后的映射，否则用 embedding 的 linear 映射回去
#             self.register_buffer("w_lm_head", dequantize_linear(torch_model.lm_head))
#         else:
#             self.w_lm_head = None

#         self.torch_model = torch_model

#     def forward(
#         self,
#         inputs: torch.Tensor,
#         offset: int | None = None,
#         cache: list[TinyKvCache] | None = None,
#     ) -> torch.Tensor:
#         h = self.embedding(inputs)
#         for layer in range(self.num_hidden_layers):
#             h = self.layers_inner[layer](
#                 h, mask="causal", offset=offset, cache=cache[layer]
#             )
#         h = self.norm(h)
#         if self.w_lm_head is not None:
#             return linear(h, self.w_lm_head)
#         else:
#             return self.embedding.as_linear(h)
