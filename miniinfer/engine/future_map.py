"""
FutureMap - 循环 Buffer 管理异步结果

用于双 Batch Overlap 场景：
- GPU 执行 Batch B forward 时，Batch A 的结果已存储在 FutureMap
- CPU 可以从 FutureMap 读取 Batch A 结果进行 sampling/后处理

Future Placeholder 机制：
- Decode batch 的 input_ids 使用负数索引作为占位符：-(future_index + 1)
- 在 forward_stream 上，forward 开始前调用 resolve_future_input_ids() 替换占位符
- 这允许 schedule(N) 与 forward(N-1) 并行执行

参考 sglang 的 FutureMap 实现，使用循环 buffer 避免频繁分配。
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class FutureIndices:
  """FutureMap 中分配的槽位索引"""

  indices: torch.Tensor  # [batch_size] 每个 seq 的槽位
  interval: slice  # 连续区间 slice(start, end)


class FutureMap:
  """
  循环 Buffer 存储异步采样结果

  布局:
  - [0, future_limit): 活跃区域，用于存储待处理的 token IDs
  - 使用 future_ct 追踪当前分配位置（模运算实现循环）

  生命周期:
  1. alloc_future_indices(bs) -> 分配 bs 个槽位，返回 placeholder indices
  2. store_to_map(indices, result) -> GPU 写入结果（在 forward_stream 上）
  3. resolve_future_input_ids(input_ids) -> 替换占位符（在 forward_stream 上，forward 前）
  """

  def __init__(
    self,
    max_running_requests: int,
    max_chunks_per_request: int = 4,
    device: str = "cuda",
  ):
    """
    初始化 FutureMap

    Args:
        max_running_requests: 最大并发请求数
        max_chunks_per_request: 每个请求最多 chunk 数（用于 chunked prefill）
        device: 存储设备
    """
    self.device = device

    # 计算 buffer 大小：每个请求可能产生多个 chunk 的 future
    # 预留 3x 空间以容纳双 buffer + chunked prefill
    self.future_limit = max_running_requests * (3 + max_chunks_per_request)
    # 额外预留 2x max_running_requests 防止边界 wrap-around
    self.buffer_len = self.future_limit + 2 * max_running_requests

    # 主要 buffer: 存储采样的 token IDs
    self.token_ids_buf = torch.empty((self.buffer_len,), dtype=torch.int64, device=device)

    # 可选: 存储 logits 用于调试或 speculative decoding
    # self.logits_buf = None  # 按需分配

    # 循环指针
    self.future_ct = 0

    logger.info(
      f"FutureMap initialized: buffer_len={self.buffer_len}, "
      f"future_limit={self.future_limit}, device={device}"
    )

  def alloc_future_indices(self, batch_size: int) -> FutureIndices:
    """
    为新 batch 分配 buffer 槽位

    Args:
        batch_size: 本次分配的序列数

    Returns:
        FutureIndices 包含分配的索引和区间
    """
    cur_ct = self.future_ct
    # 循环更新指针
    self.future_ct = (cur_ct + batch_size) % self.future_limit

    # 计算分配区间 [start, end)
    # 使用 1-based indexing，这样 placeholder = -(index) 永远是负数
    start = cur_ct + 1
    end = cur_ct + 1 + batch_size

    # 创建索引 tensor
    indices = torch.arange(start, end, dtype=torch.int64, device=self.device)

    return FutureIndices(indices=indices, interval=slice(start, end))

  def get_placeholder_input_ids(self, future_indices: FutureIndices) -> torch.Tensor:
    """
    获取用于 decode batch 的 placeholder input_ids

    Placeholder 使用负数表示：placeholder = -future_index
    在 forward 前调用 resolve_future_input_ids() 替换为真实 token

    Args:
        future_indices: 之前分配的槽位

    Returns:
        placeholder_ids: [batch_size] 负数占位符
    """
    return -future_indices.indices

  def store_to_map(
    self,
    future_indices: FutureIndices,
    next_token_ids: torch.Tensor,
  ) -> None:
    """
    将采样结果写入 buffer

    应该在 forward_stream 上调用，sample 之后。

    Args:
        future_indices: 之前分配的槽位
        next_token_ids: 采样的 token IDs [batch_size]
    """
    intv = future_indices.interval
    self.token_ids_buf[intv] = next_token_ids

  def resolve_future_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    """
    在 forward_stream 上替换 input_ids 中的占位符

    应该在 forward 开始前调用。
    - 负数表示占位符：input_ids[i] < 0 表示需要从 buffer 读取
    - 正数是真实 token：保持不变

    Args:
        input_ids: 可能包含占位符的 input_ids [batch_size]

    Returns:
        resolved_ids: 替换后的 input_ids [batch_size]
    """
    # 找出负数位置（占位符）
    is_placeholder = input_ids < 0

    if not is_placeholder.any():
      # 没有占位符，直接返回
      return input_ids

    # 从 buffer 读取真实 token
    # placeholder = -future_index，所以 future_index = -placeholder
    future_indices = torch.clamp(-input_ids, min=0)
    resolved_tokens = self.token_ids_buf[future_indices]

    # 替换占位符
    return torch.where(is_placeholder, resolved_tokens, input_ids)

  def resolve_future(
    self,
    future_indices: FutureIndices,
  ) -> torch.Tensor:
    """
    从 buffer 读取采样结果（兼容旧 API）

    Args:
        future_indices: 之前分配的槽位

    Returns:
        token_ids: [batch_size] 采样的 token IDs
    """
    intv = future_indices.interval
    return self.token_ids_buf[intv]

  def reset(self) -> None:
    """重置循环指针（用于新 session）"""
    self.future_ct = 0
