import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine.kv_cache import BatchingKvCache

from .tinyllm_base import *
from .utils import *


def rope_helper(deviceStr: str, traditional: bool, precision: torch.dtype):
  BATCH_SIZE = 16
  NUM_HEADS = 8
  HEAD_DIM = 4
  MAX_SEQ_LEN = 14
  SEQ_LEN = 9
  BASE = 10000

  device = torch.device("cuda") if deviceStr == "cuda" else torch.device("cpu")
  for _ in range(100):
    user_layer = RotaryEmbedding(HEAD_DIM, MAX_SEQ_LEN, BASE, traditional=traditional)
    x = torch.rand((BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=precision)

    input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN, size=BATCH_SIZE)
    input_pos_user = [slice(i, i + SEQ_LEN) for i in input_pos]

    freqs_cis = precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN, BASE).to(device)

    if traditional:
      reference_output = apply_rotary_emb(x, freqs_cis, input_pos_user)
    else:
      reference_output = apply_rotary_emb_qwen2(x, freqs_cis, input_pos_user)

    user_output = user_layer(x, input_pos_user).to(device)
    assert_allclose(
      user_output,
      reference_output,
      precision,
      atol=5e-6 if precision == torch.float32 else 1e-3,
    )


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("traditional", [False, True], ids=["default", "traditional"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rope_multiple_offsets(device: str, traditional: bool, precision: torch.dtype):
  rope_helper(device, traditional, precision)


def attention_helper(deviceStr: str, H_q, H, L, E, S, BATCH, use_flash_attention: bool = False):
  precision = torch.float32
  device = torch.device("cuda") if deviceStr == "cuda" else torch.device("cpu")

  q_shape = (BATCH, H_q, L, E)
  kv_shape = (BATCH, H, S, E)
  scale = 0.8
  for _ in range(100):
    query = torch.rand(q_shape, dtype=precision, device=device)
    key = torch.rand(kv_shape, dtype=precision, device=device)
    value = torch.rand(kv_shape, dtype=precision, device=device)
    mask = torch.rand((BATCH, 1, L, S), dtype=precision, device=device)

    reference_output_1 = ref_scaled_dot_product_attention(
      query=query,
      key=key,
      value=value,
      scale=scale,
      attn_mask=mask,
      enable_gqa=True,
    )
    reference_output_2 = ref_scaled_dot_product_attention(
      query=query,
      key=key,
      value=value,
      scale=scale,
      enable_gqa=True,
    )

    user_output_1 = scaled_dot_product_attention_grouped(
      query,
      key,
      value,
      scale=scale,
      mask=mask,
    )
    user_output_2 = scaled_dot_product_attention_grouped(
      query,
      key,
      value,
      scale=scale,
    )

    assert_allclose(
      user_output_2,
      reference_output_2,
      precision=torch.float16,
    )
    assert_allclose(
      user_output_1,
      reference_output_1,
      precision=torch.float16,
    )


def test_task_1_attention_with_mask_cpu_small():
  attention_helper("cpu", 6, 3, 2, 5, 3, 1, use_flash_attention=False)


def test_task_1_attention_with_mask_cpu():
  attention_helper("cpu", 18, 6, 7, 5, 3, 10, use_flash_attention=False)


def test_task_1_attention_with_mask_cpu_large():
  attention_helper("cpu", 28, 4, 16, 128, 16, 3, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu_extra_small():
  attention_helper("cuda", 1, 1, 5, 7, 4, 1, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu_small():
  attention_helper("cuda", 6, 3, 2, 5, 3, 1, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu():
  attention_helper("cuda", 18, 6, 7, 5, 3, 10, use_flash_attention=False)


def test_task_1_attention_with_mask_gpu_large():
  attention_helper("cuda", 28, 4, 16, 128, 16, 3, use_flash_attention=False)


def helper_test_task_3(model_name: str, seq_len: int, iters: int = 1):
  """Tests for continuous batching of decode requests."""
  requests = 4
  max_seq_len = seq_len

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  torch_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = Qwen2ModelV2(torch_model)
  for _ in range(iters):
    cache = [BatchingKvCache(requests, max_seq_len) for _ in range(model.num_hidden_layers)]
    # Start each request at a staggered token index.
    staggered_start = [seq_len * i // requests for i in range(requests)]
    inputs = torch.randint(0, tokenizer.vocab_size, (requests, seq_len), device=device)
    ref_outputs = torch_model(inputs).logits

    for offset in range(seq_len + staggered_start[-1]):
      seq_idx = [offset - start for start in staggered_start]

      # Requests join at the staggered start, and leave when they reach seq_len.
      for request_id, sidx in enumerate(seq_idx):
        if sidx == 0:
          for c in cache:
            c.add_request(TinyKvFullCache(), request_id)
        elif sidx == seq_len:
          for c in cache:
            c.remove_request(request_id)

      next_tokens = []
      next_offsets = []
      for request_id, sidx in enumerate(seq_idx):
        if 0 <= sidx < seq_len:
          next_tokens.append(inputs[request_id, sidx].item())
          next_offsets.append(sidx)
        else:  # 非活跃请求直接填0
          next_tokens.append(0)
          next_offsets.append(0)

      user_out = model(
        inputs=torch.tensor(next_tokens, dtype=torch.int32, device=device).reshape(-1, 1),
        offset=torch.tensor(next_offsets, dtype=torch.int32, device=device),
        cache=cache,
      )

      for request_id, sidx in enumerate(seq_idx):
        if 0 <= sidx < seq_len:
          user_out_r = user_out[request_id, 0, :]
          ref_out_r = ref_outputs[request_id, sidx, :]
          user_out_r = user_out_r - torch.logsumexp(user_out_r, dim=-1, keepdim=True)
          ref_out_r = ref_out_r - torch.logsumexp(ref_out_r, dim=-1, keepdim=True)
          assert_allclose(user_out_r, ref_out_r, precision=torch.float16, rtol=1e-1)


@pytest.mark.skipif(not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found")
def test_task_3_qwen_2_05b():
  helper_test_task_3("Qwen/Qwen2-0.5B-Instruct", seq_len=3)


@pytest.mark.skipif(not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct model not found")
def test_task_3_qwen_2_15b():
  helper_test_task_3("Qwen/Qwen2-1.5B-Instruct", seq_len=3)
