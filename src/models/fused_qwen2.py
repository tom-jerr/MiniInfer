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
from .configs.config_qwen2 import Qwen2Config
from triton_ops.rms_norm import add_rms_norm_forward
from triton_ops.silu_mul import silu_and_mul_forward
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

class Qwen2Attention(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
    ):
        super().__init__()
        self.config = config

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
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

        # Update cache if provided
        if past_key_value is not None:
            # TinyKvCache expects [B, H, S, D]
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            key_states, value_states, _, mask = past_key_value.update_and_fetch(
                key_states, value_states, mask_length=input_shape[1], mask=mask
            )
            # Transpose back to [B, S, H, D] for flash_attn
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        # FlashAttention supports GQA. We pass q, k, v separately.
        # q: [B, S_q, H_q, D], k: [B, S_k, H_kv, D], v: [B, S_k, H_kv, D]
        # For decoding (q_len=1), we want to attend to all past keys, so causal=False.
        # For prefill (q_len > 1), we need causal masking.
        is_causal = query_states.shape[1] > 1
        attn_output = flash_attn_func(
            query_states, key_states, value_states, softmax_scale=self.scaling, causal=is_causal
        )
        attn_output = self.o_proj(attn_output.reshape(*input_shape, -1).contiguous())

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
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        assert config.hidden_act == "silu", "Qwen2 MLP only supports SiLU activation."
        self.act_fn = get_activation(config.hidden_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP(x)=(SiLU(W_gate(x))⊙W_up(x))W_down"""
        # 1. GEMM
        gate_up = self.gate_up_proj(x)
        # 2. SiLU + Mul
        act_out = silu_and_mul_forward(gate_up)
        down_proj = self.down_proj(act_out)
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
        self.eps = config.rms_norm_eps
        self.w_post_layernorm = w_post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        past_key_value: Optional[TinyKvCache] = None,
        mask: torch.Tensor | str | None = None,
        use_cache: bool = False,
        offset: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[TinyKvCache]]:
        # 1. Input Norm + Residual Add (Fused)
        if residual is None:
            residual = hidden_states
            normed = self.input_layernorm(hidden_states)
        else:
            residual, normed = add_rms_norm_forward(
                residual, hidden_states, self.input_layernorm.weight, self.eps
            )

        # 2. Attention
        attn_output, updated_cache = self.self_attn(
            normed,
            past_key_value=past_key_value,
            mask=mask,
            use_cache=use_cache,
            offset=offset,
        )

        # 3. Post Attention Norm + Residual Add (Fused)
        residual, normed = add_rms_norm_forward(
            residual, attn_output, self.w_post_layernorm, eps=self.eps
        )

        # 4. MLP
        mlp_output = self.mlp(normed)

        # Return mlp_output and residual separately for next layer fusion
        return mlp_output, residual, updated_cache


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
            w_post_attention_layernorm = torch_model.model.layers[
                i
            ].post_attention_layernorm.weight.to(precision)

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
                layer.self_attn.q_proj.weight.copy_(
                    torch_model.model.layers[i].self_attn.q_proj.weight
                )
                layer.self_attn.k_proj.weight.copy_(
                    torch_model.model.layers[i].self_attn.k_proj.weight
                )
                layer.self_attn.v_proj.weight.copy_(
                    torch_model.model.layers[i].self_attn.v_proj.weight
                )
                layer.self_attn.o_proj.weight.copy_(
                    torch_model.model.layers[i].self_attn.o_proj.weight
                )
                layer.self_attn.q_proj.bias.copy_(torch_model.model.layers[i].self_attn.q_proj.bias)
                layer.self_attn.k_proj.bias.copy_(torch_model.model.layers[i].self_attn.k_proj.bias)
                layer.self_attn.v_proj.bias.copy_(torch_model.model.layers[i].self_attn.v_proj.bias)
                gate_weight = torch_model.model.layers[i].mlp.gate_proj.weight
                up_weight = torch_model.model.layers[i].mlp.up_proj.weight
                layer.mlp.gate_up_proj.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))
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
        hidden_states = self.embedding(inputs)
        residual = None

        # Handle backward compatibility: cache (positional) takes precedence over past_key_values (keyword)
        if cache is not None:
            past_key_values = cache

        # Initialize caches if needed
        if use_cache and past_key_values is None:
            past_key_values = [TinyKvFullCache() for _ in range(len(self.layers_inner))]

        updated_caches = [] if use_cache else None

        for idx, layer in enumerate(self.layers_inner):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, residual, updated_cache = layer(
                hidden_states,
                residual=residual,
                past_key_value=past_kv,
                mask=mask,
                use_cache=use_cache,
                offset=offset,
            )
            if use_cache:
                updated_caches.append(updated_cache)

        # Final Norm
        if residual is not None:
            _, hidden_states = add_rms_norm_forward(
                residual, hidden_states, self.norm.weight, self.norm.eps
            )
        else:
            hidden_states = self.norm(hidden_states)

        if self.w_lm_head is not None:
            logits = linear(hidden_states, self.w_lm_head)
        else:
            logits = self.embedding.as_linear(hidden_states)

        return logits, updated_caches if use_cache else None