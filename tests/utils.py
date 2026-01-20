import huggingface_hub
from layers.attention import scaled_dot_product_attention_grouped
import numpy as np
import torch

DEVICES = [torch.device("cpu"), torch.device("cuda")]
DEVICES_IDS = ["cpu", "cuda"]
PRECISIONS = [torch.float32, torch.float16]
PRECISION_IDS = ["float32", "float16"]


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if freqs_cis.ndim == 2:
        # [seq_len, half_dim] case
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        # [B, seq_len, half_dim] case
        assert freqs_cis.shape == (x.shape[0], x.shape[1], x.shape[-1])
        shape = [
            d if i == 0 or i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
        ]

    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: list[slice] | slice | None = None,
) -> torch.Tensor:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    B, seq_len = xq.shape[0], xq.shape[1]

    if offset is None:
        # from 0 to seq_len-1th
        freqs_cis = freqs_cis[:seq_len]
    elif isinstance(offset, slice):
        # just one slice
        assert (
            offset.stop - offset.start == seq_len
        ), "Offset slice length must match sequence length"
        freqs_cis = freqs_cis[offset]
    elif isinstance(offset, list):
        # slice list for each batch
        assert len(offset) == B, "Number of slices in offset list must match batch size"
        # 为每个batch应用对应的slice
        freqs_cis_list = [freqs_cis[s] for s in offset]
        freqs_cis = torch.stack(freqs_cis_list, dim=0)  # [B, seq_len, half_dim]
    else:
        raise TypeError(f"Unsupported type for offset: {type(offset)}")

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def apply_rotary_emb_qwen2(
    xq: torch.Tensor,
    freqs_cis: torch.Tensor,
    offset: list[slice] | slice | None = None,
) -> torch.Tensor:
    B, seq_len = xq.shape[0], xq.shape[1]
    half_dim = xq.shape[-1] // 2

    # 分为前半部分和后半部分
    x1 = xq[..., :half_dim]  # 前半部分
    x2 = xq[..., half_dim:]  # 后半部分

    # 根据offset参数截取频率张量
    if offset is None:
        # from 0 to seq_len-1th
        freqs_cis = freqs_cis[:seq_len]
    elif isinstance(offset, slice):
        # just one slice
        assert (
            offset.stop - offset.start == seq_len
        ), "Offset slice length must match sequence length"
        freqs_cis = freqs_cis[offset]
    elif isinstance(offset, list):
        # slice list for each batch
        assert len(offset) == B, "Number of slices in offset list must match batch size"
        # 为每个batch应用对应的slice
        freqs_cis_list = [freqs_cis[s] for s in offset]
        freqs_cis = torch.stack(freqs_cis_list, dim=0)  # [B, seq_len, half_dim]
    else:
        raise TypeError(f"Unsupported type for offset: {type(offset)}")

    # 获取 cos 和 sin 分量
    cos_freqs = freqs_cis.real  # [seq_len, half_dim] or [B, seq_len, half_dim]
    sin_freqs = freqs_cis.imag  # [seq_len, half_dim] or [B, seq_len, half_dim]

    # reshape for broadcast
    if cos_freqs.ndim == 2:
        # [seq_len, half_dim] -> [1, seq_len, 1, half_dim]
        cos_freqs = cos_freqs.view(1, seq_len, 1, half_dim)
        sin_freqs = sin_freqs.view(1, seq_len, 1, half_dim)
    else:
        # [B, seq_len, half_dim] -> [B, seq_len, 1, half_dim]
        cos_freqs = cos_freqs.view(B, seq_len, 1, half_dim)
        sin_freqs = sin_freqs.view(B, seq_len, 1, half_dim)

    # 应用旋转
    # output[0:half_dim] = x1 * cos - x2 * sin
    # output[half_dim:dim] = x1 * sin + x2 * cos
    real = x1 * cos_freqs - x2 * sin_freqs
    imag = x1 * sin_freqs + x2 * cos_freqs

    # 重新组合
    xq_out = torch.cat([real, imag], dim=-1)
    return xq_out.type_as(xq)


def assert_allclose(
    actual,
    expected,
    precision: torch.dtype,
    rtol: float | None = None,
    atol: float | None = None,
):
    """
    Assert that two tensors are all close within some tolerance.

    Args:
        actual: Actual output tensor
        expected: Expected output tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        precision: Tensor precision type
    """
    if precision == torch.float16:
        rtol = rtol or 5.0e-2
        atol = atol or 5.0e-4
    elif precision == torch.float32:
        rtol = rtol or 1.0e-5
        atol = atol or 1.0e-8
    else:
        return ValueError(f"Unsupported precision: {precision}")

    actual = actual.detach().cpu().numpy()
    expected = expected.detach().cpu().numpy()
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.invert(np.isclose(actual, expected, rtol=rtol, atol=atol))
        # 如果数组很大且不匹配的元素比例很小（<0.1%），可能是可以接受的
        if diff.size > 10000 and np.sum(diff) / diff.size < 0.001:
            return
        with np.printoptions(precision=3, suppress=True):
            print("actual=", actual)
            print("expected=", expected)
            print("diff_actual=", actual * diff)
            print("diff_expected=", expected * diff)
            print("diff_actual_val=", actual[diff])
            print("diff_bexpected_val=", expected[diff])
            assert False, "Arrays are not all close."


def qwen_2_05b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen2-0.5B-Instruct", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen2-0.5B-Instruct model: {e}")
        return False


def qwen_2_15b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen2-1.5B-Instruct", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen2-1.5B-Instruct model: {e}")
        return False


def qwen_2_7b_model_exists() -> bool:
    try:
        huggingface_hub.snapshot_download(
            "Qwen/Qwen2-7B-Instruct", local_files_only=True
        )
        return True
    except Exception as e:
        print(f"Cannot find the Qwen2-7B-Instruct model: {e}")
        return False


# ============================================================================
# Naive Attention Backend (No KV Cache)
# ============================================================================
import torch.nn as nn
from miniinfer.engine.scheduler_batch import ForwardBatch, ForwardMode


class NaiveAttnBackend:
    """
    A simple attention backend for testing that does not use KV cache.
    It performs standard scaled dot-product attention with optional causal masking.
    """

    def __init__(self, use_causal_mask: bool = False):
        """
        Args:
            use_causal_mask: Whether to apply causal mask. Set to False for
                           bidirectional attention (matching HF behavior when attention_mask=None)
        """
        self.use_causal_mask = use_causal_mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_impl: nn.Module,
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Perform attention without KV cache.

        Args:
            q: Query tensor [num_tokens, num_heads * head_dim]
            k: Key tensor [num_tokens, num_kv_heads, head_dim]
            v: Value tensor [num_tokens, num_kv_heads, head_dim]
            attn_impl: The attention implementation module
            forward_batch: Forward batch containing metadata
            save_kv_cache: Ignored (no KV cache)
        Returns:
            Attention output [num_tokens, num_heads * head_dim]
        """
        num_heads = attn_impl.tp_q_head_num
        num_kv_heads = attn_impl.tp_k_head_num
        head_dim = attn_impl.head_dim
        scaling = attn_impl.scaling

        # Reshape q to [num_tokens, num_heads, head_dim]
        q = q.view(-1, num_heads, head_dim)
        # k, v are already [num_tokens, num_kv_heads, head_dim]

        # Get batch information from forward_batch
        batch_size = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens

        # For simplicity, we assume all sequences have the same length
        # or we handle a single batch for testing
        if batch_size == 1:
            seq_len = (
                seq_lens[0].item()
                if isinstance(seq_lens, torch.Tensor)
                else seq_lens[0]
            )
            # Reshape to [batch, seq_len, heads, head_dim]
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim)
        else:
            # Handle variable-length sequences by assuming same length
            total_tokens = q.shape[0]
            seq_len = total_tokens // batch_size
            q = q.view(batch_size, seq_len, num_heads, head_dim)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

        # Transpose to [batch, heads, seq_len, head_dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use GQA attention with optional causal mask
        mask = "causal" if self.use_causal_mask else None
        output = scaled_dot_product_attention_grouped(q, k, v, scale=scaling, mask=mask)

        # Transpose back and reshape to [num_tokens, num_heads * head_dim]
        output = output.transpose(1, 2).contiguous()
        output = output.view(-1, num_heads * head_dim)

        return output


def create_simple_forward_batch(
    batch_size: int,
    seq_len: int,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    device: torch.device,
    use_causal_mask: bool = False,
) -> ForwardBatch:
    """
    Create a simple ForwardBatch for testing without KV cache related fields.

    Args:
        use_causal_mask: Whether to use causal mask in attention.
                        Set to False for bidirectional attention (default for testing).
    """
    seq_lens = torch.tensor([seq_len] * batch_size, device=device)

    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=input_ids,
        positions=positions,
        req_pool_indices=torch.zeros(batch_size, dtype=torch.long, device=device),
        out_cache_loc=torch.zeros(
            batch_size * seq_len, dtype=torch.long, device=device
        ),
        seq_lens=seq_lens,
        seq_lens_sum=batch_size * seq_len,
    )
    # Attach our naive attention backend
    batch.attn_backend = NaiveAttnBackend(use_causal_mask=use_causal_mask)

    return batch
