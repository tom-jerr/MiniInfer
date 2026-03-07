"""
Incremental Streaming Decoder - 增量流式解码器

用于流式输出的增量 detokenize。

优化策略 (基于 SGLang):
1. Offset-based 增量解码：通过 read_offset 跟踪已确认的有效字符位置，
   避免每次都从头解码，实现 O(1) 增量解码而非 O(n) 全量解码
2. LRU 缓存：缓存已解码的文本段，减少重复解码开销
3. UTF-8 边界处理：通过 valid char 检测处理 byte-fallback token 的不完整序列
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field


# Replacement character for invalid UTF-8 sequences
REPLACEMENT_CHAR = "\ufffd"


@dataclass
class DecodeState:
  """
  单个请求的解码状态

  采用 offset-based 增量解码策略:
  - read_offset: 已确认有效的字符位置（可安全输出的文本长度）
  - skip_offset: 跳过的字符数（用于处理不完整 UTF-8 序列）
  - surr_offset: 代理字符相关的偏移处理
  """

  # 已处理的所有 token ids
  token_ids: List[int] = field(default_factory=list)
  # 上一次解码后的完整文本
  text: str = ""
  # 已确认有效的字符位置 (SGLang offset-based strategy)
  read_offset: int = 0
  # 跳过的字符偏移 (处理 surrogate 和无效字符)
  surr_offset: int = 0
  # 是否已完成
  finished: bool = False
  # Whether token_ids has changed since last decode
  dirty: bool = False
  # 输出的累积文本 (已确认可输出的部分)
  output_text: str = ""


class IncrementalDecoder:
  """
  增量流式解码器

  用于将模型生成的 token ids 增量解码为文本，支持流式输出。

  使用方法:
  ```python
  decoder = IncrementalDecoder(tokenizer)

  # 每次模型生成新 token 后调用
  delta_text, finished = decoder.decode(request_id, new_token_id, eos_token_id)

  # 输出增量文本
  if delta_text:
    stream_output(delta_text)

  # 请求完成后清理状态
  if finished:
    decoder.cleanup(request_id)
  ```
  """

  def __init__(
    self,
    tokenizer: Any,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: Optional[bool] = False,
  ):
    """
    初始化增量解码器

    Args:
        tokenizer: HuggingFace tokenizer 实例
        skip_special_tokens: 解码时是否跳过特殊 token
        clean_up_tokenization_spaces: 是否清理空格（None 表示使用 tokenizer 默认值）
        on_token_callback: 可选的回调函数，签名为 (req_id, token_id, delta_text, finished)
    """
    self.tokenizer = tokenizer
    self.skip_special_tokens = skip_special_tokens
    self.clean_up_tokenization_spaces = clean_up_tokenization_spaces

    # 每个请求的解码状态
    self._states: Dict[int, DecodeState] = {}

    # EOS token id
    self._eos_token_id = getattr(tokenizer, "eos_token_id", None)

    # LRU 解码缓存（缓存 token_ids tuple -> decoded text）
    self._decode_cache_size = 1024
    self._decode_cache: Dict[int, str] = {}
    self._cache_hits = 0
    self._cache_misses = 0

  def _get_cache_key(self, token_ids: List[int]) -> int:
    """生成缓存键（使用 token_ids 的 hash）"""
    return hash(tuple(token_ids))

  def _decode_with_cache(self, token_ids: List[int]) -> str:
    """带缓存的解码"""
    cache_key = self._get_cache_key(token_ids)
    if cache_key in self._decode_cache:
      self._cache_hits += 1
      return self._decode_cache[cache_key]

    self._cache_misses += 1
    result = self._decode(token_ids)

    # 简单的 LRU 策略：超过容量时清理一半
    if len(self._decode_cache) >= self._decode_cache_size:
      # 清理最早的一半条目
      keys_to_remove = list(self._decode_cache.keys())[: self._decode_cache_size // 2]
      for k in keys_to_remove:
        del self._decode_cache[k]

    self._decode_cache[cache_key] = result
    return result

  def _decode(self, token_ids: List[int]) -> str:
    """对全部 token ids 做一次 decode（兼容不同 tokenizer 签名）"""
    kwargs = {"skip_special_tokens": self.skip_special_tokens}
    if self.clean_up_tokenization_spaces is not None:
      kwargs["clean_up_tokenization_spaces"] = self.clean_up_tokenization_spaces
    try:
      return self.tokenizer.decode(token_ids, **kwargs)
    except TypeError:
      # 兼容不支持 clean_up_tokenization_spaces 的 tokenizer
      kwargs.pop("clean_up_tokenization_spaces", None)
      return self.tokenizer.decode(token_ids, **kwargs)

  def _batch_decode(self, token_id_seqs: List[List[int]]) -> List[str]:
    """Batch decode for multiple token id sequences."""
    kwargs = {"skip_special_tokens": self.skip_special_tokens}
    if self.clean_up_tokenization_spaces is not None:
      kwargs["clean_up_tokenization_spaces"] = self.clean_up_tokenization_spaces
    batch_decode = getattr(self.tokenizer, "batch_decode", None)
    if batch_decode is None:
      return [self._decode(ids) for ids in token_id_seqs]
    try:
      return batch_decode(token_id_seqs, **kwargs)
    except TypeError:
      kwargs.pop("clean_up_tokenization_spaces", None)
      return batch_decode(token_id_seqs, **kwargs)

  @staticmethod
  def _delta_text(old_text: str, new_text: str) -> str:
    """计算新旧文本的增量（假设旧文本是新文本的前缀）"""
    if new_text.startswith(old_text):
      return new_text[len(old_text) :]
    # 回退：找最长公共前缀，尽量减少漏发
    max_len = min(len(old_text), len(new_text))
    i = 0
    while i < max_len and old_text[i] == new_text[i]:
      i += 1
    return new_text[i:]

  @staticmethod
  def _find_valid_utf8_boundary(text: str, read_offset: int) -> int:
    """
    找到有效的 UTF-8 字符边界

    检查从 read_offset 开始的文本是否包含无效字符（如 replacement char）。
    如果流末尾有不完整的 UTF-8 序列，返回安全的截断位置。

    SGLang 策略：如果末尾有 \ufffd，可能是不完整的 byte-fallback token，
    需要等待更多 token 才能确定。

    Returns:
        安全的输出位置
    """
    if read_offset >= len(text):
      return read_offset

    # 检查末尾是否有 replacement character
    new_text = text[read_offset:]
    if not new_text:
      return read_offset

    # 如果末尾是 replacement char，可能是不完整的序列
    # 需要等待更多 token
    if new_text.endswith(REPLACEMENT_CHAR):
      # 找到最后一个非 replacement char 的位置
      safe_end = len(new_text) - 1
      while safe_end > 0 and new_text[safe_end - 1] == REPLACEMENT_CHAR:
        safe_end -= 1
      return read_offset + safe_end

    return len(text)

  def _incremental_decode(self, state: DecodeState, final: bool = False) -> str:
    """
    Offset-based 增量解码 (SGLang 策略)

    追踪已确认安全输出的文本位置，只输出新增加的部分。
    处理 byte-fallback token 导致的不完整 UTF-8 序列。

    Args:
        state: 解码状态
        final: 是否为最后一次解码（EOS 时应输出所有剩余内容）

    Returns:
        增量文本
    """
    if not state.token_ids:
      return ""

    # 解码全部 token
    full_text = self._decode_with_cache(state.token_ids)
    state.text = full_text
    state.dirty = False

    if final:
      # 最后一次解码，输出所有剩余内容
      delta = full_text[state.read_offset :]
      state.read_offset = len(full_text)
      state.output_text += delta
      return delta

    # 找到安全的输出边界
    safe_end = self._find_valid_utf8_boundary(full_text, state.read_offset)

    if safe_end <= state.read_offset:
      return ""  # 还没有新的安全字符可以输出

    # 计算增量文本
    delta = full_text[state.read_offset : safe_end]
    state.read_offset = safe_end
    state.output_text += delta

    return delta

  def get_or_create_state(self, req_id: int) -> DecodeState:
    """获取或创建请求的解码状态"""
    if req_id not in self._states:
      self._states[req_id] = DecodeState()
    return self._states[req_id]

  def decode(
    self,
    req_id: int,
    token_id: int,
    eos_token_id: Optional[int] = None,
  ) -> Tuple[str, bool]:
    """
    对单个 token 进行增量解码

    Args:
        req_id: 请求 ID
        token_id: 新生成的 token id
        eos_token_id: 结束符 token id（可选，不传则使用 tokenizer 的默认值）

    Returns:
        Tuple[str, bool]: (增量文本, 是否完成)
    """
    state = self.get_or_create_state(req_id)

    if state.finished:
      return "", True

    # 规范化 token_id 为 int，兼容 list/tuple/tensor 形式
    if isinstance(token_id, (list, tuple)):
      if len(token_id) == 0:
        return "", False
      token_id = token_id[0]
    if hasattr(token_id, "item"):
      token_id = int(token_id.item())

    # 确定 EOS token id
    eos_id = eos_token_id if eos_token_id is not None else self._eos_token_id

    # 检查是否是 EOS token
    is_eos = token_id == eos_id

    # 添加新 token
    state.token_ids.append(token_id)
    state.dirty = True

    # 使用 offset-based 增量解码
    delta_text = self._incremental_decode(state, final=is_eos)

    # 更新状态
    if is_eos:
      state.finished = True

    return delta_text, is_eos

  def decode_batch(
    self,
    req_ids: List[int],
    token_ids: List[int],
    eos_token_id: Optional[int] = None,
    return_full_text: bool = False,
  ) -> List[Any]:
    """
    批量增量解码

    Args:
        req_ids: 请求 ID 列表
        token_ids: 对应的新生成 token id 列表
        eos_token_id: 结束符 token id

    Returns:
        List: 每个请求的 (增量文本, 是否完成) 列表；当 return_full_text=True 时，
              返回 (增量文本, 是否完成, full_text)。
    """
    eos_id = eos_token_id if eos_token_id is not None else self._eos_token_id

    states: List[Optional[DecodeState]] = []
    old_read_offsets: List[int] = []  # 记录更新前的 read_offset
    is_eos_flags: List[bool] = []
    token_ids_int: List[int] = []

    for req_id, token_id in zip(req_ids, token_ids):
      state = self.get_or_create_state(req_id)
      if state.finished:
        states.append(None)
        old_read_offsets.append(0)
        is_eos_flags.append(True)
        token_ids_int.append(-1)
        continue

      if isinstance(token_id, (list, tuple)):
        if len(token_id) == 0:
          states.append(state)
          old_read_offsets.append(state.read_offset)
          is_eos_flags.append(False)
          token_ids_int.append(-1)
          continue
        token_id = token_id[0]
      if hasattr(token_id, "item"):
        token_id = int(token_id.item())
      else:
        token_id = int(token_id)

      is_eos = token_id == eos_id

      states.append(state)
      old_read_offsets.append(state.read_offset)  # 保存更新前的 offset
      is_eos_flags.append(is_eos)
      token_ids_int.append(token_id)

      state.token_ids.append(token_id)
      state.dirty = True

    to_decode_states: List[DecodeState] = [s for s in states if s is not None and s.dirty]
    to_decode_is_eos: List[bool] = []
    for s, is_eos in zip(states, is_eos_flags):
      if s is not None and s.dirty:
        to_decode_is_eos.append(is_eos)

    if to_decode_states:
      decoded_texts = self._batch_decode([s.token_ids for s in to_decode_states])
      for s, full_text, is_final in zip(to_decode_states, decoded_texts, to_decode_is_eos):
        s.text = full_text
        s.dirty = False
        # 使用 offset-based 策略计算安全输出边界
        if is_final:
          # EOS 时输出所有剩余内容
          safe_end = len(full_text)
        else:
          safe_end = self._find_valid_utf8_boundary(full_text, s.read_offset)

        if safe_end > s.read_offset:
          delta = full_text[s.read_offset : safe_end]
          s.output_text += delta
          s.read_offset = safe_end

    results: List[Any] = []
    for req_id, state, old_offset, is_eos, tid in zip(
      req_ids, states, old_read_offsets, is_eos_flags, token_ids_int
    ):
      if state is None:
        if return_full_text:
          results.append(("", True, ""))
        else:
          results.append(("", True))
        continue

      full_text = state.text
      # 计算这次的增量文本（从上一次输出位置到当前安全位置）
      delta_text = (
        full_text[old_offset : state.read_offset] if state.read_offset > old_offset else ""
      )

      if is_eos:
        state.finished = True
      if return_full_text:
        results.append((delta_text, is_eos, full_text))
      else:
        results.append((delta_text, is_eos))

    return results

  def flush(self, req_id: int) -> str:
    """
    刷新请求的所有待输出文本

    用于请求完成时，确保所有文本都被输出。

    Args:
        req_id: 请求 ID

    Returns:
        剩余的待输出文本
    """
    if req_id not in self._states:
      return ""

    state = self._states[req_id]

    # 使用 _incremental_decode with final=True 输出所有剩余内容
    return self._incremental_decode(state, final=True)

  def get_full_text(self, req_id: int) -> str:
    """
    获取请求的完整解码文本

    Args:
        req_id: 请求 ID

    Returns:
        完整的解码文本
    """
    if req_id not in self._states:
      return ""

    state = self._states[req_id]
    return state.text

  def get_output_tokens(self, req_id: int) -> List[int]:
    """
    获取请求已生成的所有 token ids

    Args:
        req_id: 请求 ID

    Returns:
        token id 列表
    """
    if req_id not in self._states:
      return []
    return self._states[req_id].token_ids.copy()

  def cleanup(self, req_id: int):
    """
    清理请求的解码状态

    Args:
        req_id: 请求 ID
    """
    if req_id in self._states:
      del self._states[req_id]

  def cleanup_all(self):
    """清理所有解码状态"""
    self._states.clear()

  def is_finished(self, req_id: int) -> bool:
    """
    检查请求是否已完成

    Args:
        req_id: 请求 ID

    Returns:
        是否已完成
    """
    if req_id not in self._states:
      return False
    return self._states[req_id].finished

  def get_stats(self) -> Dict[str, Any]:
    """
    获取解码器统计信息

    Returns:
        包含统计信息的字典
    """
    active_count = sum(1 for s in self._states.values() if not s.finished)
    finished_count = sum(1 for s in self._states.values() if s.finished)
    total_tokens = sum(len(s.token_ids) for s in self._states.values())

    cache_total = self._cache_hits + self._cache_misses
    cache_hit_rate = self._cache_hits / cache_total if cache_total > 0 else 0.0

    return {
      "active_requests": active_count,
      "finished_requests": finished_count,
      "total_tokens_decoded": total_tokens,
      "cache_size": len(self._decode_cache),
      "cache_hits": self._cache_hits,
      "cache_misses": self._cache_misses,
      "cache_hit_rate": cache_hit_rate,
    }

  def clear_cache(self):
    """清理解码缓存"""
    self._decode_cache.clear()
    self._cache_hits = 0
    self._cache_misses = 0


class ChatTemplateHandler:
  """
  Chat template 处理器

  处理 chat template 的应用和消息格式化，支持：
  1. HuggingFace tokenizer 的 apply_chat_template
  2. 自定义 Jinja2 模板
  3. 多模态消息的特殊处理

  SGLang 风格：分离文本和多模态消息处理
  """

  def __init__(
    self,
    tokenizer: Any,
    default_system_prompt: Optional[str] = None,
  ):
    """
    初始化 chat template 处理器

    Args:
        tokenizer: HuggingFace tokenizer 实例
        default_system_prompt: 默认的系统提示词
    """
    self.tokenizer = tokenizer
    self.default_system_prompt = default_system_prompt

    # 检测 tokenizer 是否支持 chat template
    self._has_chat_template = hasattr(tokenizer, "apply_chat_template")
    self._chat_template = getattr(tokenizer, "chat_template", None)

  @property
  def has_chat_template(self) -> bool:
    """检查是否支持 chat template"""
    return self._has_chat_template

  def apply_chat_template(
    self,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    return_tensors: Optional[str] = None,
    **kwargs,
  ) -> Union[str, List[int]]:
    """
    应用 chat template 将对话消息转换为模型输入

    Args:
        messages: 对话消息列表，每个消息包含 "role" 和 "content"
                 例如: [{"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi!"}]
        add_generation_prompt: 是否在末尾添加生成提示符
        tokenize: 是否进行 tokenization
        return_tensors: 返回张量类型 ("pt", "np" 等)
        **kwargs: 传递给 tokenizer.apply_chat_template 的其他参数

    Returns:
        格式化后的字符串或 token ids
    """
    if not self._has_chat_template:
      # 回退: 使用简单的格式化方式
      formatted = self._fallback_format(messages, add_generation_prompt)
      if tokenize:
        return [int(x) for x in self.tokenizer.encode(formatted)]
      return formatted

    # 处理 messages，添加默认系统提示词（如果需要）
    processed_messages = self._process_messages(messages)

    try:
      call_kwargs = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
        **kwargs,
      }
      if return_tensors is not None:
        call_kwargs["return_tensors"] = return_tensors
      try:
        return self.tokenizer.apply_chat_template(processed_messages, **call_kwargs)
      except TypeError:
        # Some tokenizers don't accept optional kwargs like return_tensors.
        call_kwargs.pop("return_tensors", None)
        return self.tokenizer.apply_chat_template(processed_messages, **call_kwargs)
    except Exception as e:
      # 出错时回退到简单格式化
      import warnings

      warnings.warn(f"apply_chat_template failed: {e}, using fallback format")
      formatted = self._fallback_format(messages, add_generation_prompt)
      if tokenize:
        return [int(x) for x in self.tokenizer.encode(formatted)]
      return formatted

  def _process_messages(
    self,
    messages: List[Dict[str, Any]],
  ) -> List[Dict[str, Any]]:
    """
    处理消息列表，添加默认系统提示词等

    Args:
        messages: 原始消息列表

    Returns:
        处理后的消息列表
    """
    if not messages:
      return messages

    processed = list(messages)

    # 如果第一条不是 system 消息且有默认系统提示词，添加它
    if self.default_system_prompt and (not processed or processed[0].get("role") != "system"):
      processed.insert(
        0,
        {
          "role": "system",
          "content": self.default_system_prompt,
        },
      )

    return processed

  def _fallback_format(
    self,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
  ) -> str:
    """
    简单的回退格式化方式

    当 tokenizer 不支持 chat template 时使用
    """
    parts = []
    for msg in messages:
      role = msg.get("role", "user")
      content = msg.get("content", "")

      if role == "system":
        parts.append(f"System: {content}\n")
      elif role == "user":
        parts.append(f"User: {content}\n")
      elif role == "assistant":
        parts.append(f"Assistant: {content}\n")
      else:
        parts.append(f"{role}: {content}\n")

    if add_generation_prompt:
      parts.append("Assistant: ")

    return "".join(parts)

  def get_chat_template_parameters(self) -> Dict[str, Any]:
    """
    获取 chat template 的参数信息

    Returns:
        包含 template 参数的字典
    """
    if not self._chat_template:
      return {"supported": False}

    # 尝试解析 Jinja2 模板以提取参数
    params = {
      "supported": True,
      "template": self._chat_template,
      "has_system_role": (
        "system" in self._chat_template.lower() if isinstance(self._chat_template, str) else False
      ),
      "has_generation_prompt": (
        "generation" in self._chat_template.lower()
        if isinstance(self._chat_template, str)
        else False
      ),
    }

    return params

  def format_for_generation(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
  ) -> str:
    """
    将单个 prompt 格式化为生成格式

    Args:
        prompt: 用户输入
        system_prompt: 可选的系统提示词

    Returns:
        格式化后的字符串
    """
    messages = []

    sys = system_prompt or self.default_system_prompt
    if sys:
      messages.append({"role": "system", "content": sys})

    messages.append({"role": "user", "content": prompt})

    return self.apply_chat_template(
      messages,
      add_generation_prompt=True,
      tokenize=False,
    )
