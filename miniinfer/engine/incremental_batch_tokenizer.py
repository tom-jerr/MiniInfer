from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .detokenizer import ChatTemplateHandler


@dataclass(frozen=True)
class _CacheKey:
  use_chat_template: bool
  system_prompt: Optional[str]
  prompt: str


class IncrementalBatchTokenizer:
  """
  Incremental batch tokenizer for prompt encoding.

  - Batch-encodes multiple prompts to reduce Python/tokenizer overhead.
  - Optionally applies chat templates via ChatTemplateHandler.
  - Keeps a small LRU cache for repeated prompts across calls.
  """

  def __init__(
    self,
    tokenizer: Any,
    *,
    chat_template_handler: Optional[ChatTemplateHandler] = None,
    use_chat_template: Union[bool, str] = "auto",
    model_name: str = "",
    system_prompt: Optional[str] = None,
    cache_size: int = 2048,
  ) -> None:
    self.tokenizer = tokenizer
    self.chat_template_handler = chat_template_handler or ChatTemplateHandler(
      tokenizer, default_system_prompt=system_prompt
    )

    self.use_chat_template = use_chat_template
    self.model_name = model_name
    self.system_prompt = system_prompt

    self._cache_size = max(int(cache_size), 0)
    self._cache: "OrderedDict[_CacheKey, List[int]]" = OrderedDict()

  def _should_apply_chat_template(self) -> bool:
    if self.use_chat_template is True:
      return self.chat_template_handler.has_chat_template
    if self.use_chat_template is False:
      return False
    # auto
    if not self.chat_template_handler.has_chat_template:
      return False
    name = (self.model_name or "").lower()
    return ("instruct" in name) or ("chat" in name)

  @staticmethod
  def _normalize_token_ids(encoded: Any) -> List[int]:
    if isinstance(encoded, torch.Tensor):
      return [int(x) for x in encoded.flatten().tolist()]
    if isinstance(encoded, dict) and "input_ids" in encoded:
      ids = encoded["input_ids"]
      if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.flatten().tolist()]
      return [int(x) for x in ids]
    if isinstance(encoded, (list, tuple)):
      return [int(x) for x in encoded]
    raise TypeError(f"Unexpected encoded type: {type(encoded)}")

  def encode_one(self, prompt: str) -> List[int]:
    use_chat = self._should_apply_chat_template()
    key = _CacheKey(use_chat, self.system_prompt, prompt)
    if self._cache_size > 0:
      cached = self._cache.get(key)
      if cached is not None:
        self._cache.move_to_end(key)
        return list(cached)

    if use_chat:
      messages: List[Dict[str, Any]] = []
      if self.system_prompt:
        messages.append({"role": "system", "content": self.system_prompt})
      messages.append({"role": "user", "content": prompt})
      encoded = self.chat_template_handler.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
      )
      token_ids = self._normalize_token_ids(encoded)
    else:
      token_ids = [int(x) for x in self.tokenizer.encode(prompt)]

    if self._cache_size > 0:
      self._cache[key] = list(token_ids)
      self._cache.move_to_end(key)
      while len(self._cache) > self._cache_size:
        self._cache.popitem(last=False)

    return token_ids

  def encode_batch(self, prompts: Sequence[str]) -> List[List[int]]:
    if not prompts:
      return []

    use_chat = self._should_apply_chat_template()
    results: List[Optional[List[int]]] = [None] * len(prompts)
    missing_prompts: List[str] = []
    missing_indices: List[int] = []

    if self._cache_size > 0:
      for i, prompt in enumerate(prompts):
        key = _CacheKey(use_chat, self.system_prompt, prompt)
        cached = self._cache.get(key)
        if cached is not None:
          self._cache.move_to_end(key)
          results[i] = list(cached)
        else:
          missing_prompts.append(prompt)
          missing_indices.append(i)
    else:
      missing_prompts = list(prompts)
      missing_indices = list(range(len(prompts)))

    if missing_prompts:
      if use_chat:
        formatted: List[str] = []
        for prompt in missing_prompts:
          formatted.append(
            self.chat_template_handler.format_for_generation(
              prompt, system_prompt=self.system_prompt
            )
          )
        batch = self.tokenizer(
          formatted,
          add_special_tokens=False,
          padding=False,
          truncation=False,
          return_attention_mask=False,
          return_token_type_ids=False,
        )
      else:
        batch = self.tokenizer(
          list(missing_prompts),
          add_special_tokens=True,
          padding=False,
          truncation=False,
          return_attention_mask=False,
          return_token_type_ids=False,
        )

      input_ids = batch["input_ids"]
      if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

      for prompt, i, ids in zip(missing_prompts, missing_indices, input_ids):
        token_ids = [int(x) for x in ids]
        results[i] = token_ids
        if self._cache_size > 0:
          key = _CacheKey(use_chat, self.system_prompt, prompt)
          self._cache[key] = list(token_ids)
          self._cache.move_to_end(key)

      while self._cache_size > 0 and len(self._cache) > self._cache_size:
        self._cache.popitem(last=False)

    # At this point, all entries are filled.
    return [r if r is not None else [] for r in results]
