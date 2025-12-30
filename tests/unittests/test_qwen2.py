import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.model.qwen2 import Qwen2Config
from src.loader.weight import load_hf_weight
from src.models.fused_qwen2 import Qwen2Model as FusedQwen2Model

from .utils import *

def helper_qwen_test(model_name: str, iters: int = 10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  torch_model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map=device
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  config = Qwen2Config.from_pretrained(model_name)
  state_dict = load_hf_weight(model_name, device)
  model = FusedQwen2Model.from_state_dict(
    config=config,
    state_dict=state_dict,
    device=device,
    precision=torch.float16,
  )

  with torch.no_grad():
    for _ in range(iters):
      input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, 10), device=device)

      user_output = model(input_ids).logits
      user_output = user_output - torch.logsumexp(user_output, dim=-1, keepdim=True)

      ref_output = torch_model(input_ids).logits
      ref_output = ref_output - torch.logsumexp(ref_output, dim=-1, keepdim=True)

      assert_allclose(user_output, ref_output, precision=torch.float16, rtol=1e-1)


@pytest.mark.skipif(not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found")
def test_task_3_qwen_2_05b():
  helper_qwen_test("Qwen/Qwen2-0.5B-Instruct", 5)

@pytest.mark.skipif(not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct model not found")
def test_task_3_qwen_2_15b():
  helper_qwen_test("Qwen/Qwen2-1.5B-Instruct", 3)

@pytest.mark.skipif(
    not qwen_2_7b_model_exists(), reason="Qwen2-7B-Instruct model not found"
)
def test_task_3_qwen_2_7b():
    helper_qwen_test("Qwen/Qwen2-7B-Instruct", 1)



