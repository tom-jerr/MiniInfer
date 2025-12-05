import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.fused_qwen2 import Qwen2Model
from src.engine.kv_cache import TinyKvFullCache, BatchingKvCache
from .utils import *

# --- PyTorch 设备设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()  # 禁用梯度计算
def helper_test_task_3(model_name: str, iters: int = 10):
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = Qwen2Model(torch_model)
    torch.manual_seed(42)
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.config.num_hidden_layers)]

        input_tensor = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(1, 10),
            dtype=torch.long,
            device=device,
        )

        user_output, _ = model(input_tensor, offset=None, cache=cache, use_cache=True)
        user_output = user_output - torch.logsumexp(user_output, dim=-1, keepdim=True)

        ref_output = torch_model(input_tensor).logits
        ref_output = ref_output - torch.logsumexp(ref_output, dim=-1, keepdim=True)

        # 假设 assert_allclose 现在使用 torch.allclose
        assert_allclose(
            user_output, ref_output, precision=torch.float16, rtol=0.1, atol=0.5
        )


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found"
)
def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct", 5)


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct model not found"
)
def test_task_3_qwen_2_15b():
    helper_test_task_3("Qwen/Qwen2-1.5B-Instruct", 3)


@torch.no_grad()  # 禁用梯度计算
def helper_test_task_4(model_name: str, seq_len: int, iters: int = 1):
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = Qwen2Model(torch_model)
    torch.manual_seed(42)
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.config.num_hidden_layers)]

        input_tensor = torch.randint(
            low=0,
            high=tokenizer.vocab_size,
            size=(1, seq_len),
            dtype=torch.long,
            device=device,
        )

        ref_outputs = torch_model(input_tensor)
        for offset in range(seq_len):
            user_output, _ = model(
                input_tensor[:, offset : offset + 1],
                offset=slice(offset, offset + 1),
                cache=cache,
                use_cache=True,
            )
            user_output = user_output.squeeze(
                1
            )  # (B, 1, vocab_size) -> (B, vocab_size)
            user_output = user_output - torch.logsumexp(
                user_output, dim=-1, keepdim=True
            )

            ref_output = ref_outputs.logits[:, offset, :]
            ref_output = ref_output - torch.logsumexp(ref_output, dim=-1, keepdim=True)

            assert_allclose(user_output, ref_output, precision=torch.float16, rtol=0.1)


@pytest.mark.skipif(
    not qwen_2_05b_model_exists(), reason="Qwen2-0.5B-Instruct model not found"
)
def test_task_4_qwen_2_05b():
    helper_test_task_4("Qwen/Qwen2-0.5B-Instruct", seq_len=3)


@pytest.mark.skipif(
    not qwen_2_15b_model_exists(), reason="Qwen2-1.5B-Instruct model not found"
)
def test_task_4_qwen_2_15b():
    helper_test_task_4("Qwen/Qwen2-1.5B-Instruct", seq_len=3)
