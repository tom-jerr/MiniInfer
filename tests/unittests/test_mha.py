# test_attention_cuda.py
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from miniinfer.layers.attention import (
    SimpleMultiHeadAttention,
    get_attention,
)
from miniinfer.layers.linear import linear, softmax
from tests.utils import *


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_softmax(device: torch.device, precision: torch.dtype):
    BATCH_SIZE, DIM = 10, 10
    for _ in range(100):
        x = torch.rand(BATCH_SIZE, DIM, dtype=precision, device=device)
        user_output = softmax(x, axis=-1)
        reference_output = F.softmax(x, dim=-1)
        assert_allclose(user_output, reference_output, precision=precision)  # type: ignore


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention(
    device: torch.device, precision: torch.dtype, batch_dimension: int
):
    if batch_dimension == 0:
        BATCH_SIZE = ()
    elif batch_dimension == 1:
        BATCH_SIZE = (2, 3)
    elif batch_dimension == 2:
        BATCH_SIZE = (2, 3, 3)
    L, D = 4, 5
    for _ in range(100):
        query = torch.rand(*BATCH_SIZE, L, D, dtype=precision, device=device)  # type: ignore
        key = torch.rand(*BATCH_SIZE, L, D, dtype=precision, device=device)  # type: ignore
        value = torch.rand(*BATCH_SIZE, L, D, dtype=precision, device=device)  # type: ignore
        reference_out = F.scaled_dot_product_attention(
            query=query.reshape(1, -1, L, D),
            key=key.reshape(1, -1, L, D),
            value=value.reshape(1, -1, L, D),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ).reshape(
            *BATCH_SIZE,
            L,
            D,  # type: ignore
        )
        user_attn = get_attention("simple")
        user_out = user_attn(query, key, value)
        assert_allclose(user_out, reference_out, precision=precision)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_linear(device: torch.device, precision: torch.dtype):
    B, DIM_X, DIM_Y = 10, 12, 10
    for _ in range(100):
        x = torch.rand(B, DIM_X, dtype=precision, device=device)
        w = torch.rand(DIM_Y, DIM_X, dtype=precision, device=device)
        b = torch.rand(DIM_Y, dtype=precision, device=device)
        user_output = linear(x, w, b)
        ref_output = torch.addmm(b, x, w.T)
        assert_allclose(user_output, ref_output, precision=precision)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(
    device: torch.device, precision: torch.dtype
):
    N, L, H, D = 10, 11, 3, 9
    embed_dim = H * D
    for _ in range(100):
        query = torch.rand(N, L, embed_dim, dtype=precision, device=device)
        key = torch.rand(N, L, embed_dim, dtype=precision, device=device)
        value = torch.rand(N, L, embed_dim, dtype=precision, device=device)

        q_proj_weight = torch.rand(embed_dim, embed_dim, dtype=precision, device=device)
        k_proj_weight = torch.rand(embed_dim, embed_dim, dtype=precision, device=device)
        v_proj_weight = torch.rand(embed_dim, embed_dim, dtype=precision, device=device)
        out_proj_weight = torch.rand(
            embed_dim, embed_dim, dtype=precision, device=device
        )

        # Reference
        reference_mha = nn.MultiheadAttention(
            embed_dim, H, batch_first=True, bias=False
        ).to(device, precision)
        reference_mha.in_proj_weight.data = torch.cat(
            [q_proj_weight, k_proj_weight, v_proj_weight], dim=0
        )
        reference_mha.out_proj.weight.data = out_proj_weight

        ref_out, _ = reference_mha(query, key, value)

        # User
        user_mha = SimpleMultiHeadAttention(
            embed_dim,
            H,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            out_proj_weight,
        ).to(device, precision)
        user_out = user_mha(query, key, value)

        assert_allclose(user_out, ref_out, precision=precision)
