"""Qwen3-MoE 模型实现（融合 QKV / MoE 专家）。

参照 ``nano-vllm/nanovllm/models/qwen3_moe.py``，适配 MiniInfer 的
``WeightLoaderMixin`` / 合并 state_dict 加载模式：

- attention 路径**复用** ``fused_qwen3.Qwen3Attention``（Qwen3-MoE 的
  attention 与 dense Qwen3 完全一致：qkv 融合、QK-Norm、attention_bias=False）；
- MLP 路径按 ``first_k_dense_replace`` / ``decoder_sparse_step`` /
  ``mlp_only_layers`` 决定该层用 dense MLP（``Qwen3MLP``）还是 MoE
  （``Qwen3MoeSparseMoeBlock`` → ``FusedMoE``）；
- per-expert HF 权重（``mlp.experts.{e}.{gate|up|down}_proj.weight``）由
  ``Qwen3MoeSparseMoeBlock.load_weights`` 调 ``FusedMoE.load_expert_weight``
  路由进堆叠权重；router 权重 ``mlp.gate.weight`` 单独 copy。

注意：``_merge_state_dict`` 已跳过 ``.experts.`` 键，故 state_dict 中
per-expert 权重保持原样，可直接按专家 id 取用。
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from miniinfer.config.model.qwen3_moe import Qwen3MoeConfig
from miniinfer.layers import (
  LMHead,
  RMSNorm,
  VocabEmbedding,
)
from miniinfer.layers.fused_moe import FusedMoE, MoEParallelConfig
from miniinfer.loader.weight import WeightLoaderMixin
from miniinfer.scheduler.scheduler_batch import ForwardBatch

from .base import BaseModelOutput
from .fused_qwen3 import Qwen3Attention, Qwen3MLP


class Qwen3MoeSparseMoeBlock(nn.Module, WeightLoaderMixin):
  """MoE 块：router gate + FusedMoE。

  把 ``gate`` 暴露在 ``mlp.gate`` 路径上（HF 键 ``mlp.gate.weight`` 需解析
  到这里）。FusedMoE 内部真正的 gate 也叫 ``experts.gate``——若不 alias，
  HF 的 ``mlp.gate.weight`` 会找不到落点，router 停在随机初始化，topk 选
  随机专家，decode 出垃圾 token。
  """

  def __init__(
    self,
    config: Qwen3MoeConfig,
    parallel: MoEParallelConfig,
    dtype: torch.dtype,
    device: torch.device,
  ):
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.norm_topk_prob = config.norm_topk_prob
    self.experts = FusedMoE(
      num_experts=config.num_experts,
      top_k=config.num_experts_per_tok,
      hidden_size=config.hidden_size,
      intermediate_size=config.moe_intermediate_size,
      norm_topk_prob=self.norm_topk_prob,
      parallel=parallel,
      dtype=dtype,
      device=device,
    )
    # router gate 别名到 mlp.gate，使 HF 键 mlp.gate.weight 能解析到这里。
    self.gate = self.experts.gate

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype):
    # router gate：HF 键 mlp.gate.weight
    gate_key = f"{prefix}.gate.weight"
    if gate_key in state_dict:
      with torch.no_grad():
        self.gate.weight.copy_(self._to_device(state_dict[gate_key], device, dtype))

    # per-expert 权重：mlp.experts.{e}.{gate|up|down}_proj.weight
    for expert_id in range(self.num_experts):
      for which in ("gate_proj", "up_proj", "down_proj"):
        key = f"{prefix}.experts.{expert_id}.{which}.weight"
        if key in state_dict:
          self.experts.load_expert_weight(
            expert_id, which,
            self._to_device(state_dict[key], device, dtype),
            param_name="weight",
          )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.experts(hidden_states)


class Qwen3MoeDecoderLayer(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen3MoeConfig,
    layer_id: int,
    parallel: MoEParallelConfig,
    dtype: torch.dtype,
    device: torch.device,
  ):
    super().__init__()
    self.hidden_size = config.hidden_size

    # 是否为 MoE 层：前 first_k_dense_replace 层用 dense；显式 mlp_only_layers
    # 用 dense；其余按 decoder_sparse_step 周期决定。
    mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
    decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
    is_dense = (
      layer_id < getattr(config, "first_k_dense_replace", 0)
      or layer_id in mlp_only_layers
      or not (
        getattr(config, "num_experts", 0) > 0
        and (layer_id + 1) % decoder_sparse_step == 0
      )
    )
    self.is_moe = not is_dense

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

    if self.is_moe:
      self.mlp = Qwen3MoeSparseMoeBlock(config, parallel, dtype, device)
    else:
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


class Qwen3MoeModel(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen3MoeConfig,
    device: torch.device,
    precision: torch.dtype = torch.float16,
    parallel: Optional[MoEParallelConfig] = None,
  ):
    super().__init__()
    self.precision = precision
    self.config = config
    self.device = device
    self.parallel = parallel or MoEParallelConfig()

    self.embedding = VocabEmbedding(
      vocab_size=config.vocab_size,
      embedding_dim=config.hidden_size,
    ).to(device)

    self.layers = nn.ModuleList()
    for layer_id in range(config.num_hidden_layers):
      layer = (
        Qwen3MoeDecoderLayer(config, layer_id, self.parallel, precision, device)
        .to(device)
        .to(precision)
      )
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


class Qwen3MoeForCausalLM(nn.Module, WeightLoaderMixin):
  def __init__(
    self,
    config: Qwen3MoeConfig,
    device: torch.device,
    precision: torch.dtype = torch.float16,
  ):
    super().__init__()
    self.precision = precision
    self.config = config
    self.device = device

    self.qwen3_moe = Qwen3MoeModel(
      config=config,
      device=device,
      precision=precision,
    ).to(device)

    self.lm_head = LMHead(vocab_size=config.vocab_size, embedding_dim=config.hidden_size).to(device)

  def load_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
    device, dtype = self.device, self.precision

    self.qwen3_moe.load_weights(state_dict)

    # tie_word_embeddings 时 lm_head 与 embed_tokens 共享权重；
    # 但 HF 仍会导出 lm_head.weight，优先使用它（LMHead.load_weights 内部已处理 tie 回退）。
    self.lm_head.load_weights(state_dict, "lm_head", device, dtype)

  @torch.no_grad()
  def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    return_hidden_states: bool = False,
  ) -> BaseModelOutput:
    qwen_out = self.qwen3_moe(
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
