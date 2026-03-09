"""
TokenPool - GPU 常驻 Token 存储

参考 mini-sglang 的 TableManager.token_pool 实现。
核心思想：
- token_pool 是一个 GPU 上的 2D tensor，shape=(max_running_reqs, max_seq_len)
- 每个请求占用一行，通过 table_idx 索引
- Prefill 阶段：将 input_ids 写入 token_pool[table_idx, 0:input_len]
- Decode 阶段：
  - input_ids = token_pool[table_indices, positions] (GPU gather，无需 H2D)
  - 采样后 token_pool[table_indices, new_positions] = next_tokens (GPU writeback)

这消除了 FutureMap placeholder → resolve 的复杂机制，
decode 的 input_ids 完全在 GPU 上获取。
"""

import torch
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class TokenPool:
  """GPU 常驻 Token 存储池"""

  def __init__(
    self,
    max_running_reqs: int,
    max_seq_len: int,
    device: str = "cuda",
  ):
    """
    初始化 TokenPool

    Args:
        max_running_reqs: 最大并发请求数
        max_seq_len: 单个请求的最大序列长度
        device: GPU 设备
    """
    self.max_running_reqs = max_running_reqs
    self.max_seq_len = max_seq_len
    self.device = device

    # table_idx 分配器：空闲槽位列表
    self._free_slots = list(range(max_running_reqs))

    # 核心数据结构：GPU 常驻 token 存储
    # shape: (max_running_reqs, max_seq_len), dtype: int32
    # 初始化为 0（padding token）
    self.token_pool = torch.zeros(
      (max_running_reqs, max_seq_len),
      dtype=torch.int32,
      device=device,
    )

    # Pinned CPU buffer 用于异步 H2D 传输
    # 与单个请求最大长度匹配即可（prefill 时批量传输）
    self.pinned_buffer = torch.zeros(
      max_seq_len,
      dtype=torch.int32,
      pin_memory=True,
    )

    logger.info(
      f"TokenPool initialized: max_running_reqs={max_running_reqs}, "
      f"max_seq_len={max_seq_len}, device={device}, "
      f"size={(max_running_reqs * max_seq_len * 4) / 1024 / 1024:.2f} MB"
    )

  @property
  def available_size(self) -> int:
    """可用槽位数量"""
    return len(self._free_slots)

  def allocate(self) -> int:
    """
    分配一个 table_idx 槽位

    Returns:
        分配的 table_idx

    Raises:
        RuntimeError: 无可用槽位
    """
    if not self._free_slots:
      raise RuntimeError("TokenPool exhausted: no free slots available")
    return self._free_slots.pop()

  def free(self, table_idx: int) -> None:
    """释放一个 table_idx 槽位"""
    if table_idx < 0 or table_idx >= self.max_running_reqs:
      raise ValueError(f"Invalid table_idx: {table_idx}")
    self._free_slots.append(table_idx)

  def write_prefill(
    self,
    table_idx: int,
    offset: int,
    token_ids: torch.Tensor,
    stream: torch.cuda.Stream = None,
  ) -> None:
    """
    Prefill 阶段：将 token_ids 写入 token_pool

    Args:
        table_idx: 目标行索引
        offset: 写入起始位置（cached_len，支持 chunked prefill）
        token_ids: 要写入的 token ids (CPU tensor, 会自动传输到 GPU)
        stream: CUDA stream (用于 non_blocking 传输)
    """
    num_tokens = len(token_ids)
    if offset + num_tokens > self.max_seq_len:
      raise RuntimeError(
        f"write_prefill overflow: offset={offset}, num_tokens={num_tokens}, "
        f"max_seq_len={self.max_seq_len}"
      )

    # 确保 token_ids 是 int32
    if token_ids.dtype != torch.int32:
      token_ids = token_ids.to(torch.int32)

    # 使用 pinned buffer 进行异步传输
    if token_ids.is_cpu:
      self.pinned_buffer[:num_tokens].copy_(token_ids.view(-1))
      src = self.pinned_buffer[:num_tokens]
    else:
      src = token_ids

    # 写入 token_pool (non_blocking)
    with torch.cuda.stream(stream) if stream else torch.cuda.device(self.device):
      self.token_pool[table_idx, offset : offset + num_tokens].copy_(
        src.to(self.device, non_blocking=True) if src.is_cpu else src
      )

  def gather_input(
    self,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
  ) -> torch.Tensor:
    """
    Decode/Extend 阶段：从 token_pool gather input_ids

    这是纯 GPU 操作，无需任何 H2D 传输。

    Args:
        table_indices: 请求的 table_idx 数组, shape=(batch_size,) 或对应 extend 展开
        positions: 每个 token 在序列中的位置, shape 与 table_indices 匹配

    Returns:
        input_ids: gather 的结果, shape 与 positions 相同
    """
    # 确保在 GPU 上
    if table_indices.device.type == "cpu":
      table_indices = table_indices.to(self.device, non_blocking=True)
    if positions.device.type == "cpu":
      positions = positions.to(self.device, non_blocking=True)

    # 对于 decode，positions = device_len - 1 (单个 token)
    # 对于 extend，positions = [cached_len, cached_len+1, ..., device_len-1] (多个 token)
    return self.token_pool[table_indices, positions]

  def write_output(
    self,
    table_indices: torch.Tensor,
    positions: torch.Tensor,
    next_tokens: torch.Tensor,
  ) -> None:
    """
    采样后：将 next_tokens 写回 token_pool

    这是纯 GPU 操作，在 compute_stream 上执行。

    Args:
        table_indices: 请求的 table_idx 数组, shape=(batch_size,)
        positions: 写入位置 (device_len), shape=(batch_size,)
        next_tokens: 采样的 token ids, shape=(batch_size,)
    """
    # 确保在 GPU 上且类型正确
    if next_tokens.dtype != torch.int32:
      next_tokens = next_tokens.to(torch.int32)

    self.token_pool[table_indices, positions] = next_tokens

  def make_input_mapping(
    self,
    reqs,
    device: torch.device,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建 input_mapping: (table_indices, positions) 用于 gather

    参考 mini-sglang _make_input_tuple()

    对于 decode 请求 (extend_len=1):
        table_indices[i] = req.table_idx
        positions[i] = req.device_len - 1  (最后一个 token 位置)

    对于 extend 请求 (extend_len>1):
        对每个 token 展开:
        table_indices[offset + j] = req.table_idx
        positions[offset + j] = req.cached_len + j

    Args:
        reqs: 请求列表，每个需要有 table_idx, cached_len, device_len 属性
        device: 目标设备

    Returns:
        (table_indices, positions): 两个 GPU tensor
    """
    # 计算总 token 数
    total_tokens = sum(self._get_extend_len(r) for r in reqs)

    # 使用 pinned memory 构建 CPU tensor
    table_indices_host = torch.empty(total_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(total_tokens, dtype=torch.int64, pin_memory=True)

    offset = 0
    for req in reqs:
      extend_len = self._get_extend_len(req)
      cached_len = self._get_cached_len(req)
      table_idx = self._get_table_idx(req)

      # 填充 table_indices
      table_indices_host[offset : offset + extend_len].fill_(table_idx)

      # 填充 positions: [cached_len, cached_len+1, ..., cached_len+extend_len-1]
      torch.arange(
        cached_len,
        cached_len + extend_len,
        dtype=torch.int64,
        out=positions_host[offset : offset + extend_len],
      )

      offset += extend_len

    # 异步传输到 GPU
    return (
      table_indices_host.to(device, non_blocking=True),
      positions_host.to(device, non_blocking=True),
    )

  def make_write_mapping(
    self,
    reqs,
    device: torch.device,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建 write_mapping: (table_indices, positions) 用于 writeback

    参考 mini-sglang _make_write_tuple()

    只为可 decode 的请求写入 next_token。
    对于 chunked prefill 的中间 chunk，不写回（没有采样）。

    Args:
        reqs: 请求列表
        device: 目标设备

    Returns:
        (table_indices, positions): 两个 GPU tensor
    """
    table_indices_list = []
    positions_list = []

    for req in reqs:
      # 只有可 decode 的请求才需要写回
      # -1 表示不写回（chunked prefill 中间 chunk）
      if self._can_decode(req):
        table_indices_list.append(self._get_table_idx(req))
        positions_list.append(self._get_device_len(req))  # 写入位置 = device_len
      else:
        # ChunkedReq 中间 chunk 不采样，写入 -1 作为占位
        table_indices_list.append(self._get_table_idx(req))
        positions_list.append(-1)

    table_indices_host = torch.tensor(table_indices_list, dtype=torch.int64, pin_memory=True)
    positions_host = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True)

    return (
      table_indices_host.to(device, non_blocking=True),
      positions_host.to(device, non_blocking=True),
    )

  # ============ Helper methods for different Req types ============

  def _get_table_idx(self, req) -> int:
    """获取 table_idx，兼容 Req 和 ChunkedReq"""
    if hasattr(req, "table_idx"):
      return req.table_idx
    elif hasattr(req, "req") and hasattr(req.req, "table_idx"):
      return req.req.table_idx
    else:
      raise AttributeError(f"Cannot get table_idx from {type(req)}")

  def _get_cached_len(self, req) -> int:
    """获取 cached_len"""
    if hasattr(req, "cached_len"):
      return req.cached_len
    elif hasattr(req, "req"):
      # ChunkedReq: 使用自己的 cached_len
      return getattr(req, "cached_len", 0)
    return 0

  def _get_device_len(self, req) -> int:
    """获取 device_len"""
    if hasattr(req, "device_len"):
      return req.device_len
    elif hasattr(req, "req") and hasattr(req.req, "device_len"):
      return req.req.device_len
    # 兼容旧 Req: 使用 current_seq_len
    if hasattr(req, "current_seq_len"):
      return req.current_seq_len
    return 0

  def _get_extend_len(self, req) -> int:
    """获取 extend_len = device_len - cached_len"""
    if hasattr(req, "extend_len"):
      return req.extend_len
    # ChunkedReq
    if hasattr(req, "chunk_size"):
      return req.chunk_size
    # 从 device_len 和 cached_len 计算
    return self._get_device_len(req) - self._get_cached_len(req)

  def _can_decode(self, req) -> bool:
    """判断请求是否可以 decode（产生输出 token）"""
    # ChunkedReq 中间 chunk 不采样
    if hasattr(req, "is_last_chunk") and not req.is_last_chunk:
      return False
    if hasattr(req, "can_decode"):
      return req.can_decode
    return True

  def reset(self) -> None:
    """重置 TokenPool 状态（用于测试）"""
    self._free_slots = list(range(self.max_running_reqs))
    self.token_pool.zero_()
