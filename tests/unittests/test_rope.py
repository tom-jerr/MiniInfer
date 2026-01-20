import numpy as np
import pytest
import torch

from miniinfer import RotaryEmbedding
from tests.utils import *


def RotaryEmbedding_helper(
    deviceStr: str, traditional: bool, precision: torch.dtype, with_offset: bool
):
    BATCH_SIZE = 1
    NUM_HEADS = 8
    HEAD_DIM = 4
    MAX_SEQ_LEN = 20
    SEQ_LEN = 10
    BASE = 10000

    device = torch.device("cuda") if deviceStr == "cuda" else torch.device("cpu")
    for _ in range(100):
        user_layer = RotaryEmbedding(
            HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional
        ).to(device)

        x = torch.rand(
            BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, device=device, dtype=precision
        )

        if with_offset:
            input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
            input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
        else:
            input_pos_user = None
            input_pos = 0

        freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN, BASE).to(device)

        if traditional:
            reference_output = apply_rotary_emb(x, freqs_cis, input_pos_user)
        else:
            reference_output = apply_rotary_emb_qwen2(x, freqs_cis, input_pos_user)

        user_output = user_layer(x, input_pos_user).to(device)

        atol = 5e-6 if precision == torch.float32 else 1e-3
        assert_allclose(user_output, reference_output, precision, atol=atol)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_torch_traditional(device, with_offset, precision):
    RotaryEmbedding_helper(device, True, precision, with_offset)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize(
    "with_offset", [True, False], ids=["with_offset", "without_offset"]
)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_rope_torch_non_traditional(device, with_offset, precision):
    RotaryEmbedding_helper(device, False, precision, with_offset)
