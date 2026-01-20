import pytest
import torch

from miniinfer import (
    Qwen2Attention as UserQwen2Attention,
)
from miniinfer import (
    Qwen2Config as UserConfig,
)
from miniinfer import (
    causal_mask,
    get_attention,
)

from ..utils import *


def grouped_attention_helper(
    device: torch.device,
    dtype: torch.dtype,
    batch_dimension: int,
    scale: float | None,
    is_causal_mask: bool,
):
    H_q = 18
    H = 6
    L = 3
    D = 5
    S = 7
    BATCH = 10
    BATCH_2 = 2

    if batch_dimension == 0:
        q_shape = (H_q, L, D)
        kv_shape = (H, S, D)
        mask_shape = (H_q, L, S)
    elif batch_dimension == 1:
        q_shape = (BATCH, H_q, L, D)
        kv_shape = (BATCH, H, S, D)
        mask_shape = (BATCH, H_q, L, S)
    elif batch_dimension == 2:
        q_shape = (BATCH_2, BATCH, H_q, L, D)
        kv_shape = (BATCH_2, BATCH, H, S, D)
        mask_shape = (BATCH_2, BATCH, H_q, L, S)

    for _ in range(100):
        query = torch.rand(q_shape, dtype=dtype, device=device)
        key = torch.rand(kv_shape, dtype=dtype, device=device)
        value = torch.rand(kv_shape, dtype=dtype, device=device)
        mask = torch.rand(mask_shape, dtype=dtype, device=device)

        # PyTorch's scaled_dot_product_attention expects shape: (batch, num_heads, seq_len, head_dim)
        # Need to reshape and transpose appropriately
        query_reshaped = query.reshape(-1, H_q, L, D)
        key_reshaped = key.reshape(-1, H, S, D)
        value_reshaped = value.reshape(-1, H, S, D)
        mask_reshaped = mask.reshape(-1, H_q, L, S) if not is_causal_mask else None

        scale_factor = scale if scale is not None else (1.0 / (D**0.5))

        # Convert mask to attention mask format (additive mask with -inf for masked positions)
        if not is_causal_mask:
            attn_mask = mask_reshaped
        else:
            attn_mask = None

        reference_output = get_attention("ref")(
            query_reshaped,
            key_reshaped,
            value_reshaped,
            enable_gqa=True,
            attn_mask=attn_mask,
            scale=scale_factor,
            is_causal=is_causal_mask,
        )

        # Reshape reference output back to original shape
        reference_output = reference_output.reshape(query.shape)

        user_output = get_attention("gqa")(
            query,
            key,
            value,
            scale=scale,
            mask=mask if not is_causal_mask else "causal",
        )

        assert_allclose(user_output, reference_output, precision=dtype)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_1_grouped_attention(
    device: str, dtype: torch.dtype, batch_dimension: int, scale: float | None
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    grouped_attention_helper(dev, dtype, batch_dimension, scale, False)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_task_2_mask_only_same_dim(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    L = 3
    S = 3
    user_output = causal_mask(L, S, torch.float32, device=dev)

    expected = torch.tensor(
        [
            [0, -float("inf"), -float("inf")],
            [0, 0, -float("inf")],
            [0, 0, 0],
        ],
        dtype=torch.float32,
        device=dev,
    )

    assert_allclose(user_output, expected, precision=torch.float32)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_task_2_mask_only_different_dim(device: str):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    L = 3
    S = 5
    user_output = causal_mask(L, S, torch.float32, device=dev)

    expected = torch.tensor(
        [
            [0, 0, 0, -float("inf"), -float("inf")],
            [0, 0, 0, 0, -float("inf")],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
        device=dev,
    )

    assert_allclose(user_output, expected, precision=torch.float32)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8])
def test_task_2_grouped_attention_causal_mask(
    device: str, dtype: torch.dtype, batch_dimension: int, scale: float | None
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)
    grouped_attention_helper(dev, dtype, batch_dimension, scale, True)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("dtype", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_3_qwen2_grouped_query_attention(
    device: str, dtype: torch.dtype, mask: str | None
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dev = torch.device(device)

    batch_size = 1
    seq_len = 4
    hidden_size = 32
    num_heads = 4
    num_kv_heads = 2
    max_seq_len = 64
    theta = 10000

    # Note: You'll need to adapt the Qwen2 model import for PyTorch
    # This is a placeholder - adjust based on your actual implementation
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2RotaryEmbedding,
    )

    config = Qwen2Config(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        rope_theta=theta,
        max_position_embeddings=max_seq_len,
    )

    config._attn_implementation = "sdpa"  # 这里可以使用causal mask
    rotary_emb = Qwen2RotaryEmbedding(config)
    torch_attention = Qwen2Attention(config, layer_idx=0).to(device=dev, dtype=dtype)

    torch.manual_seed(42)
    x = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype, device=dev) * 2 - 1
    position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)
    position_embeddings = rotary_emb(x, position_ids)

    torch_output = torch_attention(
        hidden_states=x,
        position_embeddings=position_embeddings,
        attention_mask=None,
        is_causal=mask == "causal",
    )[0]

    user_config = UserConfig(
        hidden_size=hidden_size,
        num_hidden_layers=2,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        rope_theta=theta,
        max_position_embeddings=max_seq_len,
    )

    user_attention = UserQwen2Attention(user_config)

    # 复制权重从transformers模型到用户模型
    user_attention.q_proj.weight.data = torch_attention.q_proj.weight.data.clone()
    user_attention.k_proj.weight.data = torch_attention.k_proj.weight.data.clone()
    user_attention.v_proj.weight.data = torch_attention.v_proj.weight.data.clone()
    user_attention.o_proj.weight.data = torch_attention.o_proj.weight.data.clone()
    user_attention.q_proj.bias.data = torch_attention.q_proj.bias.data.clone()
    user_attention.k_proj.bias.data = torch_attention.k_proj.bias.data.clone()
    user_attention.v_proj.bias.data = torch_attention.v_proj.bias.data.clone()

    user_attention = user_attention.to(device=dev, dtype=dtype)

    user_output = user_attention(x, mask=mask)[0]

    assert_allclose(user_output, torch_output, precision=dtype)
