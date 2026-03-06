import pytest
import torch

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


@pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="CUDA is required for overlap placeholder path",
)
def test_overlap_generate_real_model_uses_placeholder(model_path, monkeypatch):
  monkeypatch.setenv("HF_HUB_OFFLINE", "1")
  monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

  engine = LLMEngine(
    model=str(model_path),
    enable_overlap=True,
    enforce_eager=True,
    gpu_memory_utilization=0.8,
    max_num_seqs=8,
    max_context_len=512,
    use_chat_template=False,
    debug=False,
  )

  used_placeholder: list[bool] = []
  original_run_batch_async = engine.overlap_executor.run_batch_async

  def _wrapped_run_batch_async(batch, forward_batch, model_runner, use_placeholder: bool = False):
    used_placeholder.append(bool(use_placeholder))
    return original_run_batch_async(
      batch,
      forward_batch,
      model_runner,
      use_placeholder=use_placeholder,
    )

  engine.overlap_executor.run_batch_async = _wrapped_run_batch_async  # type: ignore[method-assign]

  prompts = [
    "Write one short sentence about overlap scheduling.",
    "用一句话解释什么是推理引擎的 overlap。",
  ]
  sampling_params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

  try:
    results = engine.generate(prompts, sampling_params, use_tqdm=False)
  finally:
    engine.stop()

  assert len(results) == len(prompts)
  for out in results:
    assert isinstance(out.get("text"), str)
    assert isinstance(out.get("token_ids"), list)
    assert len(out["token_ids"]) == sampling_params.max_tokens

  assert any(used_placeholder), "Expected overlap placeholder path to be used at least once"
