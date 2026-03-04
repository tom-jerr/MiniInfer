import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniinfer.config.model.qwen2 import Qwen2Config
from miniinfer.loader.weight import load_hf_weight
from miniinfer.models.fused_qwen2 import Qwen2ForCausalLM

from ..utils import *


def helper_qwen_test(model_name: str, iters: int = 10):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  torch_model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.float16, device_map=device
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  config = Qwen2Config.from_pretrained(model_name)
  model = Qwen2ForCausalLM(
    config=config,
    device=device,
    precision=torch.float16,
  )
  state_dict = load_hf_weight(model_name, device=device)
  model.load_weights(state_dict)

  with torch.no_grad():
    for _ in range(iters):
      input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, 10), device=device)
      positions = torch.arange(10, device=device).repeat(1)
      forward_batch = create_simple_forward_batch(
        batch_size=1,
        seq_len=10,
        input_ids=input_ids,
        positions=positions,
        device=device,
        use_causal_mask=True,
      )

      user_output = model(
        input_ids, positions=positions, forward_batch=forward_batch
      ).logits  # [batch * seq_len, vocab_size]
      user_output = user_output.reshape(1, 10, -1)
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
