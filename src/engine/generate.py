from collections.abc import Callable

import torch
from transformers import PreTrainedTokenizer

from ..models.qwen2 import Qwen2Model
from .kv_cache import TinyKvFullCache


def simple_generate(
  model: Qwen2Model,
  tokenizer: PreTrainedTokenizer,
  prompt: str,
  sampler: Callable[[torch.Tensor], torch.Tensor] | None = None,
  device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
  """
  Simple autoregressive text generation using greedy decoding.

  Args:
      model: The language model to use for generation
      tokenizer: Tokenizer for encoding/decoding text
      prompt: Input text prompt
      sampler: Optional sampling function (default: argmax/greedy)
      device: Device to run on
      max_new_tokens: Maximum number of new tokens to generate

  Returns:
      Generated text string untill EOS appeared
  """

  def _step(model, y):
    # Forward pass through model: y (N, S) -> output_logits (N, S, vocab_size)
    output_logits, _ = model(y)
    logits = output_logits[:, -1, :]  # (N, S, vocab_size) -> (N, vocab_size)
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # Sample next token
    if sampler is not None:
      next_token = sampler(logits)
    else:
      next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy decoding: keep (N, 1)
    return next_token

  # Setup
  model.eval().to(device)
  # Encode prompt
  tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  # Generate tokens autoregressively
  with torch.no_grad():
    while True:
      next_token = _step(model, tokens)
      tokens = torch.cat([tokens, next_token], dim=-1)
      if next_token.item() == tokenizer.eos_token_id:
        break
  print(tokenizer.decode(tokens[0], skip_special_tokens=True))


def simple_generate_with_kv_cache(
  model: Qwen2Model,
  tokenizer: PreTrainedTokenizer,
  prompt: str,
  device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
  kv_cache = [TinyKvFullCache() for _ in range(model.config.num_hidden_layers)]

  def _step(model, y, offset, cache):
    output_logits, updated_cache = model(y, offset=offset, past_key_values=cache, use_cache=True)
    logits = output_logits[:, -1, :]
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    sample = lambda x: torch.argmax(x, dim=-1, keepdim=True)
    next_token = sample(logits)
    return next_token, updated_cache

  model.eval().to(device)
  tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  output = tokens[0].tolist()  # (B, S), here B=1
  offset = 0
  with torch.no_grad():
    while True:
      next_token, kv_cache = _step(model, tokens, offset, kv_cache)
      if next_token.item() == tokenizer.eos_token_id:
        break
      output.append(next_token.item())
      # The first iteration of this loop is prefill. We want to add the offset to the prefilled token size.
      # Otherwise, we add the decoded token size (which is always 1).
      offset += tokens.size(-1)
      tokens = next_token
  print(tokenizer.decode(output, skip_special_tokens=True))


# def speculative_generate(
#     draft_model: Qwen2ModelWeek2,
#     model: Qwen2ModelWeek2,
#     draft_tokenizer: PreTrainedTokenizer,
#     tokenizer: PreTrainedTokenizer,
#     prompt: str,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
# ) -> str:
#     """
#     简化版 speculative decoding: 先用 draft model 生成若干 token，
#     再用 main model 验证并采纳部分结果。
#     """
#     draft_model.eval().to(device)
#     model.eval().to(device)

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     generated = input_ids

#     with torch.no_grad():
#         for _ in range(128):
#             # draft model 预测若干 token
#             draft_outputs = draft_model(generated, use_cache=True)
#             draft_logits = draft_outputs.logits[:, -1, :]
#             draft_next = torch.argmax(draft_logits, dim=-1)

#             # main model 验证
#             main_outputs = model(generated, use_cache=True)
#             main_logits = main_outputs.logits[:, -1, :]

#             # 如果两个模型预测一致，就采纳
#             if draft_next.item() == torch.argmax(main_logits, dim=-1).item():
#                 next_token = draft_next
#             else:
#                 next_token = torch.argmax(main_logits, dim=-1)

#             generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

#             if next_token.item() == tokenizer.eos_token_id:
#                 break

#     return tokenizer.decode(generated[0], skip_special_tokens=True)
