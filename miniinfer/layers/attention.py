from __future__ import annotations
import math
from collections.abc import Callable

import torch
import torch.nn as nn
from typing import TYPE_CHECKING
from .linear import linear, softmax
if TYPE_CHECKING:
    from engine.scheduler_batch import ForwardBatch


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the scaled dot-product attention.

    q dot k表示query和key的相似度，除以sqrt(dk)是为了防止点积过大导致softmax梯度过小
    每个query对所有key的相似度转换为一个概率分布，再用这个概率分布对value加权求和，得到最终的注意力输出

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, depth).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, depth).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, depth_v).
        scale (float, optional): Scaling factor. If None, defaults to sqrt(depth). Default is None.
        mask (torch.Tensor, optional): Mask tensor broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k). Default is None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len_q, depth_v).
    """
    q_k = torch.matmul(
        query, key.transpose(-2, -1)
    )  # (batch_size, num_heads, seq_len_q, seq_len_k)
    scores = q_k / (
        scale
        if scale is not None
        else torch.sqrt(torch.tensor(query.shape[-1], dtype=query.dtype))
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    # 每个 query 对所有 key 的相似度 转换为一个概率分布
    attn_weights = softmax(
        scores, axis=-1
    )  # (batch_size, num_heads, seq_len_q, seq_len_k)
    output = torch.matmul(
        attn_weights, value
    )  # (batch_size, num_heads, seq_len_q, depth_v)
    return output


class SimpleMultiHeadAttention(nn.Module):
    """A simple implementation of Multi-Head Attention mechanism.

    E is hidden_size or embed_dim or dims or model_dim
    H is num_heads
    D is head_dim
    L is seq_len, in PyTorch API it's S (source len)

    w_q/w_k/w_v: (H x D) x E
    output/input: N x L x E
    w_o: E x (H x D)
    拼接后的 (N, L, H*D) 还要再映射回原始维度 (N, L, E)，这样才能跟残差连接 (residual connection) 对齐。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**0.5
        assert wq.shape == (hidden_size, num_heads * self.head_dim)
        assert wk.shape == (hidden_size, num_heads * self.head_dim)
        assert wv.shape == (hidden_size, num_heads * self.head_dim)
        assert wo.shape == (num_heads * self.head_dim, hidden_size)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)  # (N, L, H, D)
            .transpose(1, 2)  # (N, H, L, D), 每个头单独计算注意力
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        x = scaled_dot_product_attention_simple(
            projection_q, projection_k, projection_v, self.scale, mask
        )  # (N, H, L, D)
        x = x.transpose(1, 2).reshape(
            N, L, self.num_heads * self.head_dim
        )  # (N, L, H*D)
        return linear(x, self.wo)  # (N, L, E)


def causal_mask(L: int, S: int, dtype: torch.dtype, device) -> torch.Tensor:
    """
    PyTorch: 如果 S > L(即在推理阶段KV cache 累积的 key 更多),
    那么 query 只能看到前 L 个 key, 这是不正确的
    [[1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0]]

    Ours:
    [[1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1]]
    """
    offset = S - L
    mask = torch.tril(torch.ones((L, S), device=device), diagonal=offset)
    mask = torch.where(mask.bool(), 0.0, -torch.inf).to(dtype=dtype, device=device)
    return mask


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """GQA: num_kv_heads << num_query_heads

    Args:
        query (torch.Tensor): [B, H_q, S_q, D]
        key (torch.Tensor): [B, H_kv, S_k, D]
        value (torch.Tensor): [B, H_kv, S_k, D]
        scale (float, optional): Scaling factor. If None, defaults to sqrt(depth). Default is None.
        mask (torch.Tensor, optional): [B, H_q, S_q, D]

    Returns:
        torch.Tensor: [B, H_q, S, D]
    """
    expect_shape = query.shape
    H_q, S_q, D = query.shape[-3:]
    H_kv, S_k, _ = key.shape[-3:]
    scale_factor = scale if scale is not None else 1.0 / (D**0.5)
    B = query.shape[:-3]
    assert H_q % H_kv == 0
    n_repeats = H_q // H_kv

    # Reshape for GQA: repeat K/V heads to match Q heads
    # query: [B, H_q, S_q, D] -> [B, H_kv, n_repeats, S_q, D]
    query = query.reshape(*B, H_kv, n_repeats, S_q, D)
    # key/value: [B, H_kv, S_k, D] -> [B, H_kv, 1, S_k, D]
    key = key.reshape(*B, H_kv, 1, S_k, D)
    value = value.reshape(*B, H_kv, 1, S_k, D)

    # 使用 float32 计算避免溢出
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()
    q_k = torch.matmul(
        query_f, key_f.transpose(-2, -1)
    )  # [B, H_kv, n_repeats, S_q, S_k]

    scores = q_k * scale_factor
    if mask is not None:
        if mask == "causal":
            mask = causal_mask(S_q, S_k, scores.dtype, scores.device)
            scores = scores + mask
        else:
            # Correct reshape: split H_q into H_kv x n_repeats without mixing batch dims
            # mask: [..., H_q, S_q, S_k] -> [..., H_kv, n_repeats, S_q, S_k]
            # This shape broadcasts correctly with scores: [B, H_kv, n_repeats, S_q, S_k]
            if len(B) == 0:
                # No batch dimension: mask is (H_q, S_q, S_k) -> (H_kv, n_repeats, S_q, S_k)
                mask = mask.reshape(H_kv, n_repeats, S_q, S_k)
            else:
                # With batch dimension: mask is (B..., H_q, S_q, S_k)
                mask = mask.broadcast_to(*B, H_q, S_q, S_k)
                mask = mask.reshape(*B, H_kv, n_repeats, S_q, S_k)
            scores = scores + mask.float()
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value_f)

    return output.reshape(expect_shape).to(query.dtype)


# Efficient implementation from pytorch
# Warning: PyTorch 官方实现的 causal mask 与我们这里的实现不同
def ref_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=S - L
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


def flash_attn_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flash Attention 包装器，适配统一接口

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, depth).
        key (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, depth).
        value (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, depth_v).
        scale (float, optional): Scaling factor. If None, defaults to sqrt(depth). Default is None.
        mask (torch.Tensor or str, optional): Mask tensor broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k) or "causal". Default is None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len_q, depth_v).
    """
    is_causal = False
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
        mask = None  # Flash Attention 内部处理 causal
    return flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scale,
        causal=is_causal,
    )


# ============================================================================
# Attention Implementation Registry
# ============================================================================

# 静态映射：字符串 -> attention实现函数
ATTENTION_IMPLEMENTATIONS: dict[str, Callable] = {
    "simple": scaled_dot_product_attention_simple,
    "gqa": scaled_dot_product_attention_grouped,  # 别名
    "ref": ref_scaled_dot_product_attention,  # 别名
    "fla": flash_attn_wrapper,  # Flash Attention
}


def get_attention(name: str) -> Callable:
    """
    根据名称获取attention实现函数

    Args:
        name: attention实现的名称，支持以下选项：
            - "simple": 简单的多头注意力实现
            - "grouped" 或 "gqa": 分组查询注意力(GQA)实现
            - "reference" 或 "ref": PyTorch参考实现

    Returns:
        对应的attention实现函数

    Raises:
        ValueError: 如果提供的名称不在支持的实现中

    Examples:
        >>> attn_fn = get_attention_implementation("gqa")
        >>> output = attn_fn(query, key, value, scale=0.125)
    """
    if name not in ATTENTION_IMPLEMENTATIONS:
        available = ", ".join(ATTENTION_IMPLEMENTATIONS.keys())
        raise ValueError(
            f"Unknown attention implementation: '{name}'. "
            f"Available options: {available}"
        )
    return ATTENTION_IMPLEMENTATIONS[name]


class AttentionImpl(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        v_head_dim: int = -1,
        use_irope: bool = False,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim != -1 else head_dim
        self.scaling = scaling
        self.layer_id = layer_id
        self.use_irope = use_irope
        self.k_scale = None
        self.v_scale = None
        self.k_scale_float = None
        self.v_scale_float = None
        self.quant_method = None

    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        **kwargs,
    ):

        k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
        v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
        return forward_batch.attn_backend.forward(
            q,
            k,
            v,
            self,
            forward_batch,
            save_kv_cache,
            **kwargs,
        )
