"""
Prefill 预算控制器

核心思路 (借鉴 SGLang):
    1. 从 KV cache 可用 token 数出发
    2. 为每个正在 decode 的请求，按 min(remaining_max_tokens, CLIP) * new_token_ratio
       精确预留空间，避免粗暴的全局比例导致过度/不足预留
    3. rem_total_tokens = available_kv - rem_total_token_offset (property，动态计算)
    4. cur_rem_tokens 跟踪当前 prefill 步骤即时可用 token
    5. 支持 chunked prefill: 超过单步预算的请求被自动截断为 chunk，
       chunked 请求必须被加入 can_run_list 以防内存泄漏
    6. 所有预算以 page_size 粒度对齐
    7. 预留足够 decode 空间，保证可调度超过最大显存的请求而不 OOM
"""

from enum import Enum, auto
from typing import Optional, List, Any
import logging

logger = logging.getLogger(__name__)

# 最大新 token 裁剪值，防止极端长输出请求过度预留 decode 空间
CLIP_MAX_NEW_TOKENS = 1024


class AddReqResult(Enum):
  """添加请求的结果"""

  CONTINUE = auto()  # 预算充足，可继续添加
  NO_TOKEN = auto()  # KV cache 不足，无法添加更多请求
  OTHER = auto()  # 其他原因（chunk 预算用尽、batch size 达到上限等）


def _align_up(value: int, alignment: int) -> int:
  """向上对齐到 alignment 的整数倍"""
  return ((value + alignment - 1) // alignment) * alignment


class PrefillAdder:
  """
  Prefill 预算控制器

  预算模型:
      available_kv_tokens     = allocator.available + cache.evictable (由外部传入)
      rem_total_token_offset  = Σ running_req_reserve + Σ prefill_consume
      rem_total_tokens        = available_kv_tokens - rem_total_token_offset
      cur_rem_token_offset    = Σ prefill_extend_tokens (仅当前批次的即时占用)
      cur_rem_tokens          = available_kv_tokens - cur_rem_token_offset

  其中 running_req_reserve = min(max_tokens - output_len, CLIP) * new_token_ratio
  """

  def __init__(
    self,
    available_kv_tokens: int,
    running_reqs: List[Any],
    new_token_ratio: float,
    max_extend_len: int,
    max_batch_size: int = 256,
    chunk_size: Optional[int] = None,
    page_size: int = 256,
    min_decode_reserve_pages: int = 2,
  ):
    """
    Args:
        available_kv_tokens: KV cache 可用 token 数 (allocator.available + cache.evictable)
        running_reqs:        正在 decode 的请求列表，用于精确计算 token 预留
        new_token_ratio:     为 decode 预留的新 token 折扣比例
        max_extend_len:      单次 prefill 最大输入 token 数
        max_batch_size:      最大 batch 大小
        chunk_size:          分块大小；None 表示禁用 chunked prefill
        page_size:           KV cache 页对齐粒度
        min_decode_reserve_pages: 为 decode 预留的最小 page 数（软限制保护）
    """
    self.max_batch_size = max_batch_size
    self.chunk_size = chunk_size
    self.page_size = page_size
    self.new_token_ratio = new_token_ratio
    self.available_kv_tokens = available_kv_tokens
    self.min_decode_reserve = min_decode_reserve_pages * page_size

    # ========== 预算偏移 ==========
    # rem_total_token_offset: 累计预留量 (running 预留 + prefill 即时占用 + prefill 新 token 预留)
    # cur_rem_token_offset:   仅当前 prefill 步的即时 extend 占用
    self.rem_total_token_offset = 0
    self.cur_rem_token_offset = 0

    # rem_input_tokens: prefill 输入 token 上限 (逐步扣减)
    self.rem_input_tokens = max_extend_len

    # rem_chunk_tokens: 当前步 chunked prefill 的剩余 chunk 预算; None = 不启用 chunked
    self.rem_chunk_tokens = chunk_size

    # ========== 为每个 running req 精确预留 decode 空间 ==========
    for req in running_reqs:
      self.rem_total_token_offset += self._running_req_token_reserve(req)

    # ========== 跟踪 ==========
    self.can_run_list: List[Any] = []
    self.new_chunked_req: Any = None  # 本轮截断产生的 chunked 请求
    self.log_input_tokens = 0
    self.log_hit_tokens = 0

    logger.debug(
      f"PrefillAdder init: available={available_kv_tokens}, "
      f"running_reserve={self.rem_total_token_offset}, "
      f"rem_total={self.rem_total_tokens}, "
      f"rem_input={self.rem_input_tokens}, "
      f"chunk_size={chunk_size}"
    )

  # ------------------------------------------------------------------ #
  #  内部工具
  # ------------------------------------------------------------------ #

  def _running_req_token_reserve(self, req: Any) -> int:
    """
    计算单个 running 请求还需要预留多少 decode 空间。

    reserve = min(max_tokens - len(output_ids), CLIP_MAX_NEW_TOKENS) * new_token_ratio
    """
    max_tokens = getattr(req, "max_tokens", 0)
    output_len = len(getattr(req, "output_ids", []))
    remaining = max(max_tokens - output_len, 0)
    clipped = min(remaining, CLIP_MAX_NEW_TOKENS)
    return int(clipped * self.new_token_ratio)

  def ceil_paged_tokens(self, tokens: int) -> int:
    """向上对齐到 page_size"""
    return _align_up(tokens, self.page_size)

  # ------------------------------------------------------------------ #
  #  预算属性（动态计算，每次调用都反映最新状态）
  # ------------------------------------------------------------------ #

  @property
  def rem_total_tokens(self) -> int:
    """扣除所有预留后的剩余 token 数（全局视角）"""
    return self.available_kv_tokens - self.rem_total_token_offset

  @property
  def cur_rem_tokens(self) -> int:
    """当前步骤的即时剩余可用 token（仅扣除本轮 extend 占用）"""
    return self.available_kv_tokens - self.cur_rem_token_offset

  @property
  def total_reqs(self) -> int:
    return len(self.can_run_list)

  @property
  def available_prefill_budget(self) -> int:
    """当前可用的 prefill token 预算"""
    return min(self.cur_rem_tokens, self.rem_input_tokens)

  # ------------------------------------------------------------------ #
  #  预算状态检查
  # ------------------------------------------------------------------ #

  def budget_state(self) -> AddReqResult:
    """检查当前预算状态"""
    if self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0:
      return AddReqResult.NO_TOKEN

    if self.rem_input_tokens <= 0:
      return AddReqResult.OTHER

    if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
      return AddReqResult.OTHER

    return AddReqResult.CONTINUE

  def no_remaining_budget(self) -> bool:
    """是否已无剩余预算"""
    return self.budget_state() != AddReqResult.CONTINUE

  # ------------------------------------------------------------------ #
  #  预算扣减
  # ------------------------------------------------------------------ #

  def _update_prefill_budget(self, prefix_len: int, extend_input_len: int, max_new_tokens: int):
    """
    扣减预算。

    Args:
        prefix_len:        prefix cache 命中长度（仅用于统计）
        extend_input_len:  对齐后的 extend 输入 token 数
        max_new_tokens:    该请求预估的最大新 token 数（完整 prefill 时传入，
                           chunked 中途传 0，因为该请求还没结束 prefill）
    """
    aligned = self.ceil_paged_tokens(extend_input_len)

    # 全局预留 += 本次 extend 占用 + 未来 decode 预留
    self.rem_total_token_offset += aligned + max_new_tokens
    # 即时占用 += 本次 extend 占用
    self.cur_rem_token_offset += aligned
    # 扣减 input / chunk 额度
    self.rem_input_tokens -= aligned
    if self.rem_chunk_tokens is not None:
      self.rem_chunk_tokens -= aligned

    self.log_hit_tokens += prefix_len
    self.log_input_tokens += aligned

  # ------------------------------------------------------------------ #
  #  Chunked prefill helper
  # ------------------------------------------------------------------ #

  def _do_chunked_prefill(
    self, req: Any, trunc_len: int, extend_len: int, max_tokens: int, reason: str
  ) -> AddReqResult:
    """
    执行 chunked prefill 截断操作。

    Args:
        req: 请求对象
        trunc_len: 截断后的长度（已对齐到 page_size）
        extend_len: 原始 extend_input_len
        max_tokens: 请求的 max_tokens（用于预留 decode 空间）
        reason: 截断原因（用于日志）

    Returns:
        AddReqResult.CONTINUE 如果成功
        AddReqResult.OTHER 如果 trunc_len <= 0
    """
    if trunc_len <= 0:
      logger.debug(f"try_add_prefill: {reason} - trunc_len={trunc_len} <= 0")
      return AddReqResult.OTHER

    # 判断是否真正截断了请求
    actually_truncated = trunc_len < extend_len

    req.extend_input_len = trunc_len
    prefix_len = len(getattr(req, "prefix_indices", []))
    req.fill_ids = req.fill_ids[: prefix_len + trunc_len]

    self.can_run_list.append(req)

    if actually_truncated:
      # 真正被截断了，标记为 chunked，不预留 decode 空间
      req.is_chunked = True
      self.new_chunked_req = req
      self._update_prefill_budget(0, trunc_len, 0)
      logger.debug(
        f"try_add_prefill: CHUNKED ({reason}) - extend_len={extend_len} "
        f"truncated to {trunc_len}, rem_total={self.rem_total_tokens}"
      )
    else:
      # 没有真正截断，完整 prefill，预留 decode 空间
      req.is_chunked = False
      self._update_prefill_budget(0, trunc_len, min(max_tokens, CLIP_MAX_NEW_TOKENS))
      logger.debug(
        f"try_add_prefill: COMPLETE ({reason}) - extend_len={extend_len}, "
        f"rem_total={self.rem_total_tokens}"
      )

    return AddReqResult.CONTINUE

  # ------------------------------------------------------------------ #
  #  添加请求
  # ------------------------------------------------------------------ #

  def try_add_prefill(self, req: Any) -> AddReqResult:
    """
    尝试添加一个新的 prefill 请求。

    检查顺序:
    1. batch size 上限
    2. 总 token 预算 (extend + decode 预留 ≤ rem_total_tokens)
       - 如果启用 chunked prefill，尝试分块处理而不是直接拒绝
    3. 输入 token 预算 (extend ≤ rem_input_tokens)
    4. chunked prefill 预算 (extend ≤ rem_chunk_tokens)

    如果 input 超过 chunk 预算但 chunk 已启用，自动截断为 chunk。

    Returns:
        AddReqResult 指示当前预算状态
    """
    # batch size 检查
    if self.total_reqs >= self.max_batch_size:
      return AddReqResult.OTHER

    extend_len = getattr(req, "extend_input_len", 0)
    if extend_len <= 0:
      return AddReqResult.CONTINUE

    max_tokens = getattr(req, "max_tokens", 0)
    output_len = len(getattr(req, "output_ids", []))
    # 该请求整体需要的 token: extend 输入 + 未来 decode 输出预留
    decode_reserve = min(max(max_tokens - output_len, 0), CLIP_MAX_NEW_TOKENS)
    total_tokens = extend_len + decode_reserve

    input_tokens = self.ceil_paged_tokens(extend_len)

    # --- 总 token 预算检查 ---
    # 修改: 当启用 chunked prefill 时，不直接返回 NO_TOKEN，
    # 而是尝试分块处理请求
    # 注意: 使用 > 而非 >=，允许 total_tokens == rem_total_tokens 的边界情况通过
    if total_tokens > self.rem_total_tokens:
      # 如果 chunked prefill 未启用，直接拒绝
      if self.rem_chunk_tokens is None:
        return AddReqResult.NO_TOKEN

      # 检查是否有足够空间进行至少一个 page 的 chunk
      min_chunk = self.page_size
      if self.rem_total_tokens < min_chunk:
        logger.debug(
          f"try_add_prefill: NO_TOKEN - rem_total_tokens={self.rem_total_tokens} "
          f"< min_chunk={min_chunk}"
        )
        return AddReqResult.NO_TOKEN

      # 在 budget_limited 分支中，total_tokens > rem_total_tokens，
      # 意味着无法完整执行请求（extend + decode 预留）。
      # 必须使用 chunked 路径：只执行一部分 extend，不预留 decode 空间。
      # chunked 请求不预留 decode，只需要保留 1 个 token 的余量
      available_for_chunk = max(self.rem_total_tokens - 1, 0)
      available_for_chunk = (available_for_chunk // self.page_size) * self.page_size

      if available_for_chunk < min_chunk:
        logger.debug(
          f"try_add_prefill: NO_TOKEN - available_for_chunk={available_for_chunk} "
          f"< min_chunk={min_chunk}"
        )
        return AddReqResult.NO_TOKEN

      # 截断为可用的 chunk 大小（取可用空间、请求长度、chunk预算的最小值）
      trunc_len = min(available_for_chunk, extend_len, self.rem_chunk_tokens or extend_len)
      trunc_len = (trunc_len // self.page_size) * self.page_size

      # 关键：既然是 budget_limited，必须确保是真正的 chunk（trunc_len < extend_len）
      # 如果 trunc_len == extend_len，说明空间足够执行完整请求，但不够 decode 预留
      # 此时必须强制截断，让出空间给 decode
      if trunc_len >= extend_len:
        # 向下减少一个 page，强制截断
        trunc_len = max(trunc_len - self.page_size, 0)
        trunc_len = (trunc_len // self.page_size) * self.page_size
        if trunc_len < min_chunk:
          logger.debug(
            f"try_add_prefill: NO_TOKEN - forced truncation resulted in "
            f"trunc_len={trunc_len} < min_chunk={min_chunk}"
          )
          return AddReqResult.NO_TOKEN

      result = self._do_chunked_prefill(req, trunc_len, extend_len, max_tokens, "budget_limited")
      if result != AddReqResult.CONTINUE:
        return AddReqResult.NO_TOKEN
      return AddReqResult.CONTINUE

    # --- 输入 token 预算检查 (已有请求时不再加，留给下轮) ---
    if input_tokens >= self.rem_input_tokens and len(self.can_run_list) > 0:
      return AddReqResult.OTHER

    # --- 分 chunked / 非 chunked 路径 ---
    if self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
      # ---- 非 chunked: 完整 prefill ----
      self.can_run_list.append(req)
      self._update_prefill_budget(
        0,
        input_tokens,
        min(max_tokens, CLIP_MAX_NEW_TOKENS),
      )
    else:
      # ---- chunked: 输入超过 chunk 预算，按 chunk_size 截断 ----
      trunc_len = (self.rem_chunk_tokens // self.page_size) * self.page_size
      result = self._do_chunked_prefill(
        req, trunc_len, extend_len, max_tokens, "chunk_size_limited"
      )
      if result != AddReqResult.CONTINUE:
        return result

    # 请求已成功加入 can_run_list，返回 CONTINUE 表示"本请求添加成功"。
    # 后续请求是否能继续添加，由调用方的 no_remaining_budget() 判断。
    # 注意：不能返回 budget_state()，因为页对齐膨胀 + decode 预留
    # 可能使 rem_total_tokens 略为负数，导致返回 NO_TOKEN，
    # 而调用方会误认为请求未被添加，形成死循环。
    return AddReqResult.CONTINUE

  def add_chunked_req(self, req: Any) -> Any:
    """
    添加续 chunk 请求（上一轮被截断的请求继续 prefill）。

    关键: chunked 请求 **必须** 被加入 can_run_list，否则已分配的 KV cache
    不会被引用到，造成内存泄漏。因此即使预算紧张也要把它加进来。

    Returns:
        req  — 如果仍被截断（还有后续 chunk）
        None — 如果本次 chunk 可以完成全部 prefill
    """
    extend_len = getattr(req, "extend_input_len", 0)

    # 可用量 = min(chunk 预算, 总预算)
    rem = self.rem_total_tokens
    if self.rem_chunk_tokens is not None:
      rem = min(rem, self.rem_chunk_tokens)
    _rem_tokens = int(rem)

    # 即使预算 ≤ 0 也必须加入列表，退而使用 chunk 预算兜底
    if _rem_tokens <= 0:
      _rem_tokens = (
        self.rem_chunk_tokens
        if self.rem_chunk_tokens is not None and self.rem_chunk_tokens > 0
        else self.page_size
      )

    truncated = extend_len > _rem_tokens
    req.extend_input_len = min(extend_len, _rem_tokens)
    prefix_len = len(getattr(req, "prefix_indices", []))
    req.fill_ids = req.fill_ids[: prefix_len + req.extend_input_len]

    self.can_run_list.append(req)

    max_tokens = getattr(req, "max_tokens", 0)
    self._update_prefill_budget(
      0,
      req.extend_input_len,
      # 仅在最后一个 chunk 时预留 decode 空间
      min(max_tokens, CLIP_MAX_NEW_TOKENS) if not truncated else 0,
    )
    req.is_chunked = truncated

    return req if truncated else None
