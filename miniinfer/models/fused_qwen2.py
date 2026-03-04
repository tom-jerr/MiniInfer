from typing import Any, Dict, Optional

from miniinfer.scheduler.scheduler_batch import ForwardBatch

from .base import BaseModelOutput
import torch
import torch.nn as nn
from kernels.triton.silu_mul import SiluAndMul
from miniinfer.layers.attention import AttentionImpl

from miniinfer.layers import (
  AttentionImpl,
  LMHead,
  RMSNorm,
  RotaryEmbedding,
  VocabEmbedding,
  get_activation,
  get_attention,
  linear,
)
from miniinfer.config.model.qwen2 import Qwen2Config
from miniinfer.loader.weight import WeightLoaderMixin


class Qwen2Attention(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: Optional[int] = None,
    layer_id: int = 0,
    rope_theta: float = 1000000,
    max_position_embeddings: int = 32768,
  ):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.num_kv_heads = num_kv_heads
    self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
    self.num_key_value_groups = num_heads // num_kv_heads
    self.scaling = self.head_dim**-0.5
    self.q_size = self.num_heads * self.head_dim
    self.kv_size = self.num_kv_heads * self.head_dim
    self.rope_theta = rope_theta
    self.max_position_embeddings = max_position_embeddings

    # 融合的 QKV 投影
    total_qkv_heads = self.num_heads + 2 * self.num_kv_heads
    self.qkv_proj = nn.Linear(self.hidden_size, total_qkv_heads * self.head_dim, bias=True)

    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    self.rope = RotaryEmbedding(
      head_size=self.head_dim,
      rotary_dim=self.head_dim,
      max_position_embeddings=self.max_position_embeddings,
      base=self.rope_theta,
      is_neox_style=True,
      dtype=torch.float16,
    )

    self.attn = AttentionImpl(
      self.num_heads,
      self.head_dim,
      self.scaling,
      num_kv_heads=self.num_kv_heads,
      layer_id=layer_id,
    )

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    qkv_w_key = f"{prefix}.qkv_proj.weight"
    qkv_b_key = f"{prefix}.qkv_proj.bias"
    o_w_key = f"{prefix}.o_proj.weight"

    with torch.no_grad():
      self.qkv_proj.weight.copy_(self._to_device(state_dict[qkv_w_key], device, dtype))
      self.qkv_proj.bias.copy_(self._to_device(state_dict[qkv_b_key], device, dtype))
      self.o_proj.weight.copy_(self._to_device(state_dict[o_w_key], device, dtype))

  def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
  ) -> torch.Tensor:
    qkv = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rope(positions, q, k)
    attn_output = self.attn(q, k, v, forward_batch)
    output = self.o_proj(attn_output)
    return output


class Qwen2MLP(nn.Module, WeightLoaderMixin):
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
    # TODO(lzy): only support silu for now
    if hidden_act != "silu":
      raise ValueError(f"Unsupported activation: {hidden_act}. " "Only silu is supported for now.")
    # SiluAndMul is a function, not a class
    self.act_fn = SiluAndMul

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    gate_up_w_key = f"{prefix}.gate_up_proj.weight"
    down_w_key = f"{prefix}.down_proj.weight"

    with torch.no_grad():
      self.gate_up_proj.weight.copy_(self._to_device(state_dict[gate_up_w_key], device, dtype))
      self.down_proj.weight.copy_(self._to_device(state_dict[down_w_key], device, dtype))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 融合投影 + 拆分
    gate_up = self.gate_up_proj(x)
    output = self.act_fn(gate_up)
    down = self.down_proj(output)
    return down


class Qwen2TransformerBlock(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen2Config,
    layer_id: int = 0,
  ):
    super().__init__()
    self.hidden_size = config.hidden_size
    rope_theta = getattr(config, "rope_theta", 1000000)
    rope_scaling = getattr(config, "rope_scaling", None)
    max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
    head_dim = getattr(config, "head_dim", None)
    self.self_attn = Qwen2Attention(
      hidden_size=config.hidden_size,
      num_heads=config.num_attention_heads,
      num_kv_heads=config.num_key_value_heads,
      head_dim=head_dim,
      layer_id=layer_id,
      rope_theta=rope_theta,
      max_position_embeddings=max_position_embeddings,
    )
    self.mlp = Qwen2MLP(
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
  ) -> torch.Tensor:
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


class Qwen2Model(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen2Config,
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
      layer = Qwen2TransformerBlock(config=config, layer_id=layer_id).to(device).to(precision)
      self.layers.append(layer)

    self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps).to(device)

  @classmethod
  def from_state_dict(
    cls,
    config: Qwen2Config,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    precision: torch.dtype = torch.float16,
  ) -> "Qwen2Model":
    model = cls(config=config, device=device, precision=precision)
    model.load_weights(state_dict)
    return model

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
        # Post-residual layer output: x2 = residual (x1) + mlp_out
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
      # attention: do not contain the last layer hidden_states
      return hidden_states, all_hidden_states

    return hidden_states


class Qwen2ForCausalLM(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen2Config,
    device: torch.device,
    precision: torch.dtype = torch.float16,
  ):
    super().__init__()
    self.precision = precision
    self.config = config
    self.device = device

    self.qwen2 = Qwen2Model(
      config=config,
      device=device,
      precision=precision,
    ).to(device)

    self.lm_head = LMHead(vocab_size=config.vocab_size, embedding_dim=config.hidden_size).to(device)

  @classmethod
  def from_state_dict(
    cls,
    config: Qwen2Config,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
    precision: torch.dtype = torch.float16,
  ) -> "Qwen2ForCausalLM":
    model = cls(config=config, device=device, precision=precision)
    model.load_weights(state_dict)
    return model

  def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
    device, dtype = self.device, self.precision

    self.qwen2.load_weights(state_dict)

    # HF causal-LM checkpoints usually store the output head as `lm_head.weight`
    # at the root, not under `model.`. We try that first for correctness.
    self.lm_head.load_weights(state_dict, "lm_head", device, dtype)

  @torch.no_grad()
  def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    return_hidden_states: bool = False,
  ) -> BaseModelOutput:
    qwen_out = self.qwen2(
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

    logits = self.lm_head(hidden_states)

    return BaseModelOutput(
      logits=logits,
      last_hidden_state=hidden_states,
      hidden_states=all_hidden_states,
    )
