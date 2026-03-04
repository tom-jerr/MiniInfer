"""
Incremental Streaming Decoder - 增量流式解码器

用于流式输出的增量 detokenize。

核心策略：每次对“全部 token ids”做一次 decode，然后用新旧文本的差异
得到增量输出，避免 byte/byte-fallback token 在单 token 解码时产生
不可逆的 U+FFFD (replacement char)。
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field


@dataclass
class DecodeState:
  """单个请求的解码状态"""

  # 已处理的所有 token ids
  token_ids: List[int] = field(default_factory=list)
  # 上一次解码后的完整文本
  text: str = ""
  # 是否已完成
  finished: bool = False


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
    on_token_callback: Optional[Callable[[int, int, str, bool], None]] = None,
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
    self.on_token_callback = on_token_callback

    # 每个请求的解码状态
    self._states: Dict[int, DecodeState] = {}

    # EOS token id
    self._eos_token_id = getattr(tokenizer, "eos_token_id", None)

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

    # 解码所有 token
    full_text = self._decode(state.token_ids)

    # 计算增量文本
    delta_text = self._delta_text(state.text, full_text)

    # 更新状态
    state.text = full_text
    if is_eos:
      state.finished = True

    if self.on_token_callback:
      self.on_token_callback(req_id, token_id, delta_text, is_eos)

    return delta_text, is_eos

  def decode_batch(
    self,
    req_ids: List[int],
    token_ids: List[int],
    eos_token_id: Optional[int] = None,
  ) -> List[Tuple[str, bool]]:
    """
    批量增量解码

    Args:
        req_ids: 请求 ID 列表
        token_ids: 对应的新生成 token id 列表
        eos_token_id: 结束符 token id

    Returns:
        List[Tuple[str, bool]]: 每个请求的 (增量文本, 是否完成) 列表
    """
    results = []
    for req_id, token_id in zip(req_ids, token_ids):
      result = self.decode(req_id, token_id, eos_token_id)
      results.append(result)
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

    full_text = self._decode(state.token_ids)
    delta = self._delta_text(state.text, full_text)
    state.text = full_text
    return delta

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

  def get_stats(self) -> Dict[str, int]:
    """
    获取解码器统计信息

    Returns:
        包含统计信息的字典
    """
    active_count = sum(1 for s in self._states.values() if not s.finished)
    finished_count = sum(1 for s in self._states.values() if s.finished)
    total_tokens = sum(len(s.token_ids) for s in self._states.values())

    return {
      "active_requests": active_count,
      "finished_requests": finished_count,
      "total_tokens_decoded": total_tokens,
    }
