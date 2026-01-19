from typing import Any, Dict

from ..layers.embedding import VocabEmbedding
from ..loader.weight import WeightLoaderMixin
from .base import BaseModelOutput
import torch
import torch.nn as nn
from ops.triton.silu_mul import SiluAndMul
from ops.triton.rms_norm import rms_norm_forward, add_rms_norm_forward

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


class Qwen2Attention(nn.Module, WeightLoaderMixin):
    def __init__(
        self,
        config: Qwen2Config,
    ):
        super().__init__()
        self.config = config

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # 融合的 QKV 投影
        total_qkv_heads = config.num_attention_heads + 2 * config.num_key_value_heads
        self.qkv_proj = nn.Linear(
            config.hidden_size, total_qkv_heads * self.head_dim, bias=True
        )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.rope = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            traditional=False,
        )

    def load_weights(
        self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype
    ):
        qkv_w_key = f"{prefix}.qkv_proj.weight"
        qkv_b_key = f"{prefix}.qkv_proj.bias"
        o_w_key = f"{prefix}.o_proj.weight"

        with torch.no_grad():
            self.qkv_proj.weight.copy_(
                self._to_device(state_dict[qkv_w_key], device, dtype)
            )
            self.qkv_proj.bias.copy_(
                self._to_device(state_dict[qkv_b_key], device, dtype)
            )
            self.o_proj.weight.copy_(
                self._to_device(state_dict[o_w_key], device, dtype)
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """Prefill-only attention (no KV cache).

        Args:
            hidden_states: (B, L, d)
            mask: (B, L, L) or "causal" or None

        Returns:
            attn_output: (B, L, E)
        """
        B, L, _ = hidden_states.shape

        # 融合 QKV 投影
        qkv = self.qkv_proj(hidden_states)

        # 拆分 Q, K, V
        q_size = self.config.num_attention_heads * self.head_dim
        kv_size = self.config.num_key_value_heads * self.head_dim

        query_states = qkv[..., :q_size]
        key_states = qkv[..., q_size : q_size + kv_size]
        value_states = qkv[..., q_size + kv_size :]

        # Reshape to multi-head: (B, L, H, D)
        query_states = query_states.view(
            B, L, self.config.num_attention_heads, self.head_dim
        )
        key_states = key_states.view(
            B, L, self.config.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            B, L, self.config.num_key_value_heads, self.head_dim
        )

        # Apply RoPE first (expects B, S, H, D format)
        query_states = self.rope(query_states)
        key_states = self.rope(key_states)

        # Then transpose to (B, H, S, D) for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Attention (prefill: S_q == S_k)
        attn_output = get_attention("gqa")(
            query_states, key_states, value_states, mask=mask, scale=self.scaling
        )

        # Output projection
        attn_output = attn_output.transpose(1, 2).reshape(B, L, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2MLP(nn.Module, WeightLoaderMixin):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # 融合的 Gate+Up 投影
        self.gate_up_proj = nn.Linear(
            self.hidden_size, 2 * self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # TODO(lzy): only support silu for now
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def load_weights(
        self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype
    ):
        gate_up_w_key = f"{prefix}.gate_up_proj.weight"
        down_w_key = f"{prefix}.down_proj.weight"

        with torch.no_grad():
            self.gate_up_proj.weight.copy_(
                self._to_device(state_dict[gate_up_w_key], device, dtype)
            )
            self.down_proj.weight.copy_(
                self._to_device(state_dict[down_w_key], device, dtype)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 融合投影 + 拆分
        gate_up = self.gate_up_proj(x)
        output = self.act_fn(gate_up)
        down = self.down_proj(output)
        return down


class Qwen2TransformerBlock(nn.Module, WeightLoaderMixin):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def load_weights(
        self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype
    ):
        self.input_layernorm.load_weights(
            state_dict, f"{prefix}.input_layernorm", device, dtype
        )
        self.post_attention_layernorm.load_weights(
            state_dict, f"{prefix}.post_attention_layernorm", device, dtype
        )
        self.self_attn.load_weights(state_dict, f"{prefix}.self_attn", device, dtype)
        self.mlp.load_weights(state_dict, f"{prefix}.mlp", device, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        """Forward pass with fused Add + RMSNorm.

        Args:
            hidden_states: (B, L, E) - current layer input
            mask: attention mask

        Returns:
            hidden_states: (B, L, E) - output after MLP residual add
        """
        # Input RMSNorm (no residual add in Qwen2)
        normed = rms_norm_forward(
            hidden_states,
            self.input_layernorm.weight,
        )

        # Attention
        attn_output = self.self_attn(normed, mask=mask)

        # 融合: attn_output + hidden_states + RMSNorm
        residual, normed = add_rms_norm_forward(
            attn_output,
            hidden_states,
            self.post_attention_layernorm.weight,
        )

        # MLP
        mlp_output = self.mlp(normed)

        # 只有 Add，没有后面的 Norm
        hidden_states = residual + mlp_output

        return hidden_states


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
        for _ in range(config.num_hidden_layers):
            layer = Qwen2TransformerBlock(config=config).to(device).to(precision)
            self.layers.append(layer)

        self.norm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps).to(device)
        self.lm_head = LMHead(
            vocab_size=config.vocab_size, embedding_dim=config.hidden_size
        ).to(device)

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
        self.lm_head.load_weights(state_dict, f"{p}lm_head", device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | str | None = "causal",
    ) -> BaseModelOutput:
        """Prefill-only forward (no KV cache).

        Args:
            input_ids: (B, L)
            mask: (B, L, L) or "causal" or None

        Returns:
            BaseModelOutput with logits and last_hidden_state
        """
        hidden_states = self.embedding(input_ids)
        all_hidden_states = [hidden_states]

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask=mask)
            all_hidden_states.append(hidden_states)

        # 最后一层后：只有 RMSNorm (没有 Add)
        hidden_states = rms_norm_forward(
            hidden_states,
            self.norm.weight,
        )
        all_hidden_states.append(hidden_states)

        # LM Head
        logits = self.lm_head.as_linear(hidden_states)

        return BaseModelOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
