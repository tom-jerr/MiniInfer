import torch
import pytest
from transformers import Qwen2Config as HFQwen2Config, Qwen2Model as HFQwen2Model
from miniinfer.models.fused_qwen2 import Qwen2Model
from miniinfer.loader.weight import _merge_state_dict
from miniinfer.config.model.qwen2 import Qwen2Config
from ..utils import assert_allclose


@pytest.fixture
def model_setup():
  # 1. 初始化配置 (使用小规模参数加速测试)
  config_args = {
    "vocab_size": 1000,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,  # 测试 GQA 逻辑
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
  }

  hf_config = HFQwen2Config(**config_args)
  custom_config = Qwen2Config(**config_args)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dtype = torch.float16

  # 2. 创建 HF 模型并获取 state_dict
  hf_model = HFQwen2Model(hf_config).to(device).to(dtype).eval()
  original_sd = hf_model.state_dict().copy()

  # 3. 权重融合：QKV 融合，Gate+Up 融合
  merged_sd = _merge_state_dict(original_sd)

  # 4. 创建自定义模型并加载融合后的权重
  custom_model = Qwen2Model(custom_config, device=device, precision=dtype).eval()
  custom_model.load_weights(merged_sd)

  return hf_model, custom_model, device


def test_qwen2_load_weights_precision(model_setup):
  hf_model, custom_model, device = model_setup

  # 构造随机输入
  batch_size = 2
  seq_len = 32
  input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

  with torch.no_grad():
    hf_output = hf_model(input_ids).last_hidden_state
    custom_output = custom_model(input_ids).last_hidden_state

  assert_allclose(custom_output, hf_output, precision=torch.float16, rtol=0.05, atol=0.005)
