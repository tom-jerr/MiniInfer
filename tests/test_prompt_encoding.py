import torch


class _DummyConfig:
  def __init__(self, model: str):
    self.model = model


class _DummyTokenizer:
  def __init__(self, *, apply_returns=None, apply_raises: Exception | None = None):
    self._apply_returns = apply_returns
    self._apply_raises = apply_raises
    self.applied = False
    self.encoded = False

  def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
    assert tokenize is True
    assert add_generation_prompt is True
    assert isinstance(messages, list)
    self.applied = True
    if self._apply_raises is not None:
      raise self._apply_raises
    return self._apply_returns

  def encode(self, text: str):
    self.encoded = True
    return [99, 100]


def _make_engine_like(tokenizer, *, model: str, use_chat_template, system_prompt: str):
  from miniinfer.engine.llm_engine import LLMEngine

  engine = LLMEngine.__new__(LLMEngine)
  engine.tokenizer = tokenizer
  engine.config = _DummyConfig(model=model)
  engine.use_chat_template = use_chat_template
  engine.system_prompt = system_prompt
  return engine


def test_encode_prompt_auto_uses_chat_template_for_instruct_models():
  tokenizer = _DummyTokenizer(apply_returns=[1, 2, 3])
  engine = _make_engine_like(
    tokenizer,
    model="Qwen/Qwen2-0.5B-Instruct",
    use_chat_template="auto",
    system_prompt="You are a helpful assistant.",
  )

  assert engine._encode_prompt("hi") == [1, 2, 3]
  assert tokenizer.applied is True
  assert tokenizer.encoded is False


def test_encode_prompt_auto_falls_back_when_apply_chat_template_fails():
  tokenizer = _DummyTokenizer(apply_returns=[1, 2, 3], apply_raises=RuntimeError("no"))
  engine = _make_engine_like(
    tokenizer,
    model="Qwen/Qwen2-0.5B-Instruct",
    use_chat_template="auto",
    system_prompt="You are a helpful assistant.",
  )

  assert engine._encode_prompt("hi") == [99, 100]
  assert tokenizer.applied is True
  assert tokenizer.encoded is True


def test_encode_prompt_handles_tensor_return_from_apply_chat_template():
  tokenizer = _DummyTokenizer(apply_returns=torch.tensor([4, 5, 6]))
  engine = _make_engine_like(
    tokenizer,
    model="Qwen/Qwen2-0.5B-Instruct",
    use_chat_template="auto",
    system_prompt="You are a helpful assistant.",
  )

  assert engine._encode_prompt("hi") == [4, 5, 6]


def test_encode_prompt_never_uses_chat_template_when_disabled():
  tokenizer = _DummyTokenizer(apply_returns=[1, 2, 3])
  engine = _make_engine_like(
    tokenizer,
    model="Qwen/Qwen2-0.5B-Instruct",
    use_chat_template=False,
    system_prompt="You are a helpful assistant.",
  )

  assert engine._encode_prompt("hi") == [99, 100]
  assert tokenizer.applied is False
  assert tokenizer.encoded is True
