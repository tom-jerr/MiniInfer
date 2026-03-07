"""
Qwen3 模型实现（融合 QKV/Gate-Up 投影）

与 Qwen2 的主要区别:
1. QKV 投影无 bias (attention_bias=False)
2. head_dim 独立配置（Qwen3-0.6B: head_dim=128, 而 hidden_size/num_heads=64）
3. QK-Norm: 在 RoPE 之前对 Q/K 进行 per-head RMSNorm
"""

from typing import Dict, Optional

from miniinfer.scheduler.scheduler_batch import ForwardBatch

from .base import BaseModelOutput
import torch
import torch.nn as nn
from kernels.triton.silu_mul import SiluAndMul
from miniinfer.layers import (
  AttentionImpl,
  LMHead,
  RMSNorm,
  RotaryEmbedding,
  VocabEmbedding,
)
from miniinfer.config.model.qwen3 import Qwen3Config
from miniinfer.loader.weight import WeightLoaderMixin


class Qwen3Attention(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    layer_id: int = 0,
    rope_theta: float = 1000000.0,
    max_position_embeddings: int = 40960,
    rms_norm_eps: float = 1e-6,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim
    self.num_key_value_groups = num_heads // num_kv_heads
    self.scaling = head_dim**-0.5
    self.q_size = num_heads * head_dim
    self.kv_size = num_kv_heads * head_dim

    # 融合的 QKV 投影（无 bias，Qwen3 的 attention_bias=False）
    total_qkv_heads = num_heads + 2 * num_kv_heads
    self.qkv_proj = nn.Linear(hidden_size, total_qkv_heads * head_dim, bias=False)
    self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    # QK-Norm: per-head RMSNorm，应用在 RoPE 之前
    self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
    self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    self.rope = RotaryEmbedding(
      head_size=head_dim,
      rotary_dim=head_dim,
      max_position_embeddings=max_position_embeddings,
      base=rope_theta,
      is_neox_style=True,
      dtype=torch.float16,
    )

    self.attn = AttentionImpl(
      num_heads,
      head_dim,
      self.scaling,
      num_kv_heads=num_kv_heads,
      layer_id=layer_id,
    )

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    qkv_w_key = f"{prefix}.qkv_proj.weight"
    o_w_key = f"{prefix}.o_proj.weight"

    with torch.no_grad():
      self.qkv_proj.weight.copy_(self._to_device(state_dict[qkv_w_key], device, dtype))
      self.o_proj.weight.copy_(self._to_device(state_dict[o_w_key], device, dtype))

    # QK-Norm 权重
    self.q_norm.load_weights(state_dict, f"{prefix}.q_norm", device, dtype)
    self.k_norm.load_weights(state_dict, f"{prefix}.k_norm", device, dtype)

  def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
  ) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

    # QK-Norm: reshape 到 per-head 后做 RMSNorm，再 reshape 回来
    # split 后的 tensor 可能不连续，用 reshape 代替 view
    # q: [T, num_heads * head_dim] → [T * num_heads, head_dim] → norm → [T, num_heads * head_dim]
    T = q.shape[0]
    q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(T, -1)
    k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(T, -1)

    q, k = self.rope(positions, q, k)
    attn_output = self.attn(q, k, v, forward_batch)
    output = self.o_proj(attn_output)
    return output


class Qwen3MLP(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
  ):
    super().__init__()
    # 融合的 Gate+Up 投影
    self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    if hidden_act != "silu":
      raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
    self.act_fn = SiluAndMul

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    gate_up_w_key = f"{prefix}.gate_up_proj.weight"
    down_w_key = f"{prefix}.down_proj.weight"

    with torch.no_grad():
      self.gate_up_proj.weight.copy_(self._to_device(state_dict[gate_up_w_key], device, dtype))
      self.down_proj.weight.copy_(self._to_device(state_dict[down_w_key], device, dtype))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate_up = self.gate_up_proj(x)
    output = self.act_fn(gate_up)
    down = self.down_proj(output)
    return down


class Qwen3TransformerBlock(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen3Config,
    layer_id: int = 0,
  ):
    super().__init__()
    self.hidden_size = config.hidden_size
    self.self_attn = Qwen3Attention(
      hidden_size=config.hidden_size,
      num_heads=config.num_attention_heads,
      num_kv_heads=config.num_key_value_heads,
      head_dim=config.head_dim,
      layer_id=layer_id,
      rope_theta=config.rope_theta,
      max_position_embeddings=config.max_position_embeddings,
      rms_norm_eps=config.rms_norm_eps,
    )
    self.mlp = Qwen3MLP(
      hidden_size=config.hidden_size,
      intermediate_size=config.intermediate_size,
      hidden_act=config.hidden_act,
    )
    self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    self.input_layernorm.load_weights(state_dict, f"{prefix}.input_layernorm", device, dtype)
    self.post_attention_layernorm.load_weights(
      state_dict, f"{prefix}.post_attention_layernorm", device, dtype
    )
    self.self_attn.load_weights(state_dict, f"{prefix}.self_attn", device, dtype)
    self.mlp.load_weights(state_dict, f"{prefix}.mlp", device, dtype)

  def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
  ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if residual is None:
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
    else:
      hidden_states, residual = self.input_layernorm(hidden_states, residual)

    hidden_states = self.self_attn(
      positions=positions,
      hidden_states=hidden_states,
      forward_batch=forward_batch,
    )

    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual


class Qwen3Model(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen3Config,
    device: torch.device,
    precision: torch.dtype = torch.float16,
  ):
    super().__init__()
    self.precision = precision
    self.config = config
    self.device = device

    self.embedding = VocabEmbedding(
      vocab_size=config.vocab_size,
      embedding_dim=config.hidden_size,
    ).to(device)

    self.layers = nn.ModuleList()
    for layer_id in range(config.num_hidden_layers):
      layer = Qwen3TransformerBlock(config=config, layer_id=layer_id).to(device).to(precision)
      self.layers.append(layer)

    self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps).to(device)

  def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
    device, dtype = self.device, self.precision
    has_model_prefix = "model.embed_tokens.weight" in state_dict
    p = "model." if has_model_prefix else ""

    self.embedding.load_weights(state_dict, f"{p}embed_tokens", device, dtype)

    for i, layer in enumerate(self.layers):
      layer.load_weights(state_dict, f"{p}layers.{i}", device, dtype)

    self.norm.load_weights(state_dict, f"{p}norm", device, dtype)

  def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    return_hidden_states: bool = False,
  ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    hidden_states = self.embedding(input_ids)
    all_hidden_states = [] if return_hidden_states else None

    residual = None
    for layer in self.layers:
      if return_hidden_states:
        all_hidden_states.append(hidden_states if residual is None else hidden_states + residual)
      hidden_states, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
      )

    if residual is not None:
      hidden_states = self.norm(hidden_states + residual)
    else:
      hidden_states = self.norm(hidden_states)

    if return_hidden_states:
      return hidden_states, all_hidden_states

    return hidden_states


class Qwen3ForCausalLM(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen3Config,
    device: torch.device,
    precision: torch.dtype = torch.float16,
  ):
    super().__init__()
    self.precision = precision
    self.config = config
    self.device = device

    self.qwen3 = Qwen3Model(
      config=config,
      device=device,
      precision=precision,
    ).to(device)

    self.lm_head = LMHead(vocab_size=config.vocab_size, embedding_dim=config.hidden_size).to(device)

  def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
    device, dtype = self.device, self.precision

    self.qwen3.load_weights(state_dict)

    # Qwen3-0.6B 使用 tie_word_embeddings=True，lm_head 与 embed_tokens 共享权重
    # 但 HF 仍会导出 lm_head.weight，优先使用它
    self.lm_head.load_weights(state_dict, "lm_head", device, dtype)

  @torch.no_grad()
  def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    return_hidden_states: bool = False,
  ) -> BaseModelOutput:
    qwen_out = self.qwen3(
      input_ids=input_ids,
      positions=positions,
      forward_batch=forward_batch,
      return_hidden_states=return_hidden_states,
    )

    if return_hidden_states:
      hidden_states, all_hidden_states = qwen_out
    else:
      hidden_states = qwen_out
      all_hidden_states = None

    # EXTEND 模式下只取每个序列最后一个 token 的 hidden state 做 lm_head，节省显存
    hidden_states_for_logits = hidden_states
    if forward_batch.forward_mode.is_extend():
      extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None) or []
      if extend_lens:
        if hidden_states.dim() == 2:
          last_token_indices = []
          cumsum = 0
          for length in extend_lens:
            last_token_indices.append(cumsum + int(length) - 1)
            cumsum += int(length)
          hidden_states_for_logits = hidden_states[
            torch.tensor(
              last_token_indices,
              device=hidden_states.device,
              dtype=torch.int64,
            )
          ]
        elif hidden_states.dim() == 3:
          bsz = hidden_states.size(0)
          idx = torch.tensor(
            [int(length) - 1 for length in extend_lens[:bsz]],
            device=hidden_states.device,
            dtype=torch.int64,
          ).clamp_min_(0)
          hidden_states_for_logits = hidden_states[torch.arange(bsz, device=idx.device), idx]

    logits = self.lm_head(hidden_states_for_logits)

    return BaseModelOutput(
      logits=logits,
      last_hidden_state=hidden_states,
      hidden_states=all_hidden_states,
    )
