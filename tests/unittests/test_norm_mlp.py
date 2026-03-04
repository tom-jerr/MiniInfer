import pytest
import torch
import torch.nn.functional as F

from miniinfer.layers import RMSNorm
from miniinfer.models.fused_qwen2 import Qwen2MLP as UserQwen2MLP

from ..utils import *


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_rms_norm(device: str, precision: torch.dtype):
  SIZE = 100
  SIZE_Y = 111
  eps = 1e-5
  for _ in range(100):  # Reduced loop for faster testing
    data = torch.rand((SIZE, SIZE_Y), device=device, dtype=precision)
    weight = torch.rand((SIZE_Y,), device=device, dtype=precision)
    user_norm = RMSNorm(dim=SIZE_Y, eps=eps).to(device=device, dtype=precision)
    user_norm.load_weights(
      state_dict={"random.weight": weight}, prefix="random", device=device, dtype=precision
    )
    user_output = user_norm(data)
    reference_output = F.rms_norm(data, normalized_shape=(SIZE_Y,), weight=weight, eps=eps)
    assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_task_1_rms_norm_cast_to_float32(device: str):
  precision = torch.float16
  SIZE, SIZE_Y = 32, 64
  eps = 1e-5

  data = torch.rand((SIZE, SIZE_Y), device=device).uniform_(-1000, 1000).to(precision)
  weight = torch.rand((SIZE_Y,), device=device).uniform_(-1000, 1000).to(precision)
  user_norm = RMSNorm(dim=SIZE_Y, eps=eps).to(device=device, dtype=precision)
  user_norm.load_weights(
    state_dict={"random.weight": weight}, prefix="random", device=device, dtype=precision
  )
  user_out = user_norm(data)
  ref_out = F.rms_norm(data, normalized_shape=(SIZE_Y,), weight=weight, eps=eps)
  assert_allclose(user_out, ref_out, precision)


# Define different dimension parameters for testing
DIM_PARAMS = [
  {"batch_size": 1, "seq_len": 5, "dim": 4, "hidden_dim": 8, "id": "small_dims"},
  {"batch_size": 2, "seq_len": 16, "dim": 32, "hidden_dim": 64, "id": "large_dims"},
  {
    "batch_size": 1,
    "seq_len": 1,
    "dim": 128,
    "hidden_dim": 256,
    "id": "single_token",
  },
]
DIM_PARAMS_IDS = [d["id"] for d in DIM_PARAMS]


@pytest.mark.parametrize("device", ["cuda"], ids=["cuda"])
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize("dims", DIM_PARAMS, ids=DIM_PARAMS_IDS)
def test_task_2_qwen_mlp(device: str, precision: torch.dtype, dims: dict):
  BATCH_SIZE, SEQ_LEN, DIM, HIDDEN_DIM = (
    dims["batch_size"],
    dims["seq_len"],
    dims["dim"],
    dims["hidden_dim"],
  )

  x = torch.rand(BATCH_SIZE, SEQ_LEN, DIM, device=device, dtype=precision)
  w_gate = torch.rand(HIDDEN_DIM, DIM, device=device, dtype=precision)
  w_up = torch.rand(HIDDEN_DIM, DIM, device=device, dtype=precision)
  w_down = torch.rand(DIM, HIDDEN_DIM, device=device, dtype=precision)

  state_dict = {
    "random.gate_up_proj.weight": torch.cat([w_gate, w_up], dim=0),
    "random.down_proj.weight": w_down,
  }

  user_mlp = UserQwen2MLP(hidden_size=DIM, intermediate_size=HIDDEN_DIM, hidden_act="silu").to(
    device=device, dtype=precision
  )
  user_mlp.load_weights(state_dict, prefix="random", device=device, dtype=precision)
  user_output = user_mlp(x)

  from transformers import Qwen2Config
  from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

  config = Qwen2Config(
    hidden_size=DIM,
    intermediate_size=HIDDEN_DIM,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,  # Not used by MLP, but required by config
    rms_norm_eps=1e-6,
    vocab_size=1000,
    rope_theta=10000.0,
    max_position_embeddings=1000,
  )

  qwen2_mlp_layer = Qwen2MLP(config).to(device=device, dtype=precision)
  qwen2_mlp_layer.eval()  # Set to evaluation mode
  with torch.no_grad():
    qwen2_mlp_layer.gate_proj.weight.copy_(w_gate)
    qwen2_mlp_layer.up_proj.weight.copy_(w_up)
    qwen2_mlp_layer.down_proj.weight.copy_(w_down)
    reference_output = qwen2_mlp_layer(x)

  assert_allclose(user_output, reference_output, precision)
