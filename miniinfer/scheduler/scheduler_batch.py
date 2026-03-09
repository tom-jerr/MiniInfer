from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, List, ClassVar
from layers.attention_backend.base_backend import AttentionBackend
import torch
from itertools import count
from copy import copy
from miniinfer.utils.sampling_params import SamplingParams


class ForwardMode(IntEnum):
  # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
  # It is also called "prefill" in common terminology.
  EXTEND = auto()
  # Decode one token.
  DECODE = auto()
  # Contains both EXTEND and DECODE when doing chunked prefill.
  MIXED = auto()
  # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
  IDLE = auto()

  def is_extend(self) -> bool:
    return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

  def is_decode(self) -> bool:
    return self == ForwardMode.DECODE

  def is_mixed(self) -> bool:
    return self == ForwardMode.MIXED


class BatchType(IntEnum):
  """Batch 类型"""

  PREFILL_ONLY = auto()
  DECODE_ONLY = auto()
  MIXED = auto()


class Req:
  # block_size = 256
  counter = count()

  def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
    self.req_id = next(Req.counter)
    self.origin_input_ids = copy(token_ids)  # 切断与外部变量的联系，让 Req 拥有这份数据的独占权
    self.output_ids = []
    # fill_ids = origin_input_ids + output_ids. Used in chunked prefill.
    self.fill_ids = []
    # keep original sampling params for downstream components (e.g., sampler)
    self.sampling_params = sampling_params
    # ============ sampling related ============
    self.temperature = sampling_params.temperature
    self.max_tokens = sampling_params.max_tokens
    self.ignore_eos = sampling_params.ignore_eos
    self.top_k = sampling_params.top_k
    self.top_p = sampling_params.top_p

    # ============ finish related ============
    self.is_retracted = False
    self.ever_retracted = False
    self.finished = False
    self.finished_reason = ""

    # ============ kv cache related ============
    self.req_pool_idx: int = (
      -1
    )  # The index in the request pool, for radix cache and kv cache management
    self.extend_input_len = 0
    self.prefix_indices: torch.Tensor = None
    # ============ radix cache related ============
    """used last_node and cache_protected_len in update radix cache"""
    self.last_node = None
    self.cache_protected_len = 0  # 已经保护的 KV cache 长度，防止被驱逐

    # ============ chunked prefill related ============
    self.is_chunked = False  # 是否是 chunked 请求
    self.chunked_prefill_len = 0  # 当前 chunk 已经处理的长度
    self.total_input_len = len(token_ids)  # 总输入长度

    # ============ TokenPool related (mini-sglang style) ============
    # table_idx: 在 TokenPool 中的行索引，由 TokenPool.allocate() 分配
    self.table_idx: int = -1
    # device_len: TokenPool 中该请求已写入的 token 数量
    # 初始值为 input 长度，每次 decode 后 +1
    self.device_len: int = len(token_ids)
    # cached_len: 已经 forward 过的长度（KV cache 已计算）
    # Prefill 后 cached_len = device_len，每次 decode 后更新
    self.cached_len: int = 0

    # ============ cached metadata for performance ============
    # 缓存当前序列长度，避免重复计算 len(origin_input_ids) + len(output_ids)
    self._cached_seq_len = len(token_ids)
    # 缓存最后的 output token，避免重复访问 output_ids[-1]
    self._cached_last_token = None

  @property
  def remaining_prefill_len(self) -> int:
    """剩余需要 prefill 的长度"""
    return self.total_input_len - self.chunked_prefill_len - self.cache_protected_len

  def is_prefill_complete(self) -> bool:
    """检查 prefill 是否完成"""
    return self.chunked_prefill_len + self.cache_protected_len >= self.total_input_len

  @property
  def current_seq_len(self) -> int:
    """获取当前序列长度（缓存）"""
    return self._cached_seq_len

  def append_output_token(self, token_id: int):
    """追加输出 token 并更新缓存"""
    self.output_ids.append(token_id)
    self._cached_seq_len += 1
    self._cached_last_token = token_id

  def get_last_token(self) -> Optional[int]:
    """获取最后一个 token（优先使用缓存）"""
    if self._cached_last_token is not None:
      return self._cached_last_token
    if len(self.output_ids) > 0:
      self._cached_last_token = self.output_ids[-1]
      return self._cached_last_token
    return None

  # ============ TokenPool related methods (mini-sglang style) ============

  @property
  def extend_len(self) -> int:
    """本次 forward 需要处理的 token 数量 = device_len - cached_len"""
    return self.device_len - self.cached_len

  @property
  def remain_len(self) -> int:
    """剩余可生成的 token 数量"""
    max_device_len = self.total_input_len + self.max_tokens
    return max_device_len - self.device_len

  @property
  def can_decode(self) -> bool:
    """是否还可以继续 decode"""
    return self.remain_len > 0 and not self.finished

  def complete_one(self) -> None:
    """
    完成一次 forward 后更新状态（mini-sglang 风格）

    - cached_len 更新为 device_len（本次已计算的 KV cache）
    - device_len += 1（下一个 token 位置）
    """
    self.cached_len = self.device_len
    self.device_len += 1


class ChunkedReq:
  """
  Chunked Prefill 请求的状态追踪

  当一个请求太大无法在一个 batch 中完成 prefill 时，
  会被拆分成多个 chunk 处理。这个类用于追踪 chunked 状态。

  注意：ChunkedReq 不会进入 decode 阶段，直到所有 chunk 都处理完成。
  """

  def __init__(
    self,
    req: Req,
    cached_len: int,
    chunk_size: int,
  ):
    """
    初始化 ChunkedReq

    Args:
        req: 原始请求
        cached_len: 已缓存的长度（prefix cache 命中）
        chunk_size: 当前 chunk 的大小
    """
    self.req = req
    self.cached_len = cached_len  # 包括 prefix cache + 已处理的 chunk
    self.chunk_size = chunk_size  # 当前 chunk 要处理的 token 数

    # 不要在这里修改 req.is_chunked！
    # add_chunked_req / _do_chunked_prefill 已经正确设置了 is_chunked 状态。
    # 最后一个 chunk 时 is_chunked = False，此时不应该被改回 True。

  @property
  def input_len(self) -> int:
    """原始输入总长度"""
    return self.req.total_input_len

  @property
  def remaining_len(self) -> int:
    """处理完当前 chunk 后，还剩余的长度"""
    return self.input_len - self.cached_len - self.chunk_size

  @property
  def is_last_chunk(self) -> bool:
    """是否是最后一个 chunk"""
    return self.remaining_len <= 0

  def get_chunk_input_ids(self) -> list[int]:
    """获取当前 chunk 的 input_ids"""
    start = self.cached_len
    end = start + self.chunk_size
    return self.req.origin_input_ids[start:end]

  def update_after_chunk(self):
    """
    处理完当前 chunk 后更新状态

    如果还有剩余，更新 cached_len 以便下一个 chunk
    """
    self.req.chunked_prefill_len = self.cached_len + self.chunk_size
    if self.is_last_chunk:
      self.req.is_chunked = False  # 完成所有 chunk


@dataclass
class ScheduledBatch:
  reqs: List[Req]
  forward_mode: ForwardMode = ForwardMode.IDLE
  device: torch.device = None
  # ============ model forward related ============
  input_ids: torch.Tensor = None
  output_ids: torch.Tensor = None
  # True when this batch's input_ids contains FutureMap placeholders (negative ids)
  # and must be resolved on the forward stream before model forward.
  uses_placeholder: bool = False

  # =========== kv cache related ============
  req_pool_indices: torch.Tensor = None
  out_cache_loc: torch.Tensor = None

  # ============ some metadata ============
  seq_lens: torch.Tensor = (
    None  # from req.fill_ids = origin_input_ids + output_ids, for forward batch k_cache len
  )
  seq_lens_cpu: Optional[torch.Tensor] = None

  # ======== extend related ========
  prefix_lens: List[int] = None
  extend_lens: List[int] = None

  # ======== chunked prefill related ========
  decoding_reqs: List[Req] = None

  @classmethod
  def init_new(cls, reqs: List[Req], device: torch.device):
    return cls(
      reqs=reqs,
      device=device,
    )

  def debug_metadata(self):
    print("ScheduledBatch metadata:")
    print(f"  forward_mode: {self.forward_mode}")
    print(f"  input_ids: {self.input_ids}")
    print(f"  output_ids: {self.output_ids}")
    print(f"  req_pool_indices: {self.req_pool_indices}")
    print(f"  out_cache_loc: {self.out_cache_loc}")
    print(f"  seq_lens: {self.seq_lens}")
    print(f"  prefix_lens: {self.prefix_lens}")
    print(f"  extend_lens: {self.extend_lens}")


@dataclass
class ForwardBatch:
  """Store all inputs of a forward pass."""

  # The forward mode
  forward_mode: ForwardMode

  # ============ model forward related ============
  # The batch size
  batch_size: int
  # The input ids
  input_ids: torch.Tensor
  # Position information
  positions: torch.Tensor = None

  # ============ kv cache related ============
  # The indices of requests in the req_to_token_pool
  req_pool_indices: torch.Tensor = None
  # The indices of output tokens in the token_to_kv_pool
  out_cache_loc: torch.Tensor = None
  attn_backend: AttentionBackend = None
  # sequences that participate in this forward step (for sampling metadata)
  all_seqs: List[Req] = None

  # ============ sampling related ============
  # Optional pre-built tensors to avoid rebuilding per-step sampling params in ModelRunner.
  sampling_temperatures: Optional[torch.Tensor] = None
  sampling_top_ps: Optional[torch.Tensor] = None
  sampling_top_ks: Optional[torch.Tensor] = None
  # CPU-side metadata to avoid GPU->CPU sync in sampling.
  sampling_max_top_k: Optional[int] = None
  # Optional device-side indices of the last token per sequence in an EXTEND/MIXED
  # logits tensor of shape [total_tokens, vocab_size]. Built on CPU and transferred
  # via pinned buffer to avoid per-step GPU allocations/copies.
  last_token_indices: Optional[torch.Tensor] = None

  # ============ Pinned Memory Buffer Pool (类级别，所有实例共享) ============
  # 用于高效的 sampling params 异步传输（SGLang 风格优化）
  _pinned_temperatures: ClassVar[Optional[torch.Tensor]] = None
  _pinned_top_ps: ClassVar[Optional[torch.Tensor]] = None
  _pinned_top_ks: ClassVar[Optional[torch.Tensor]] = None
  _pinned_extend_lens: ClassVar[Optional[torch.Tensor]] = None
  _pinned_extend_prefix_lens: ClassVar[Optional[torch.Tensor]] = None
  _pinned_last_token_indices: ClassVar[Optional[torch.Tensor]] = None
  _max_buffer_size: ClassVar[int] = 256  # 默认最大 batch size

  # ============ some metadata for k ============
  seq_lens: torch.Tensor = None
  # Pre-converted int32 version to avoid GPU dtype conversion overhead
  seq_lens_int32: Optional[torch.Tensor] = None

  # =========== TODO: Maybe remove ===========
  seq_lens_sum: int = 0
  seq_lens_cpu: Optional[torch.Tensor] = None
  # Max sequence length (computed on CPU to avoid GPU sync)
  max_seq_len: Optional[int] = None

  # ============ some metadata for q
  # extend_num_tokens: Optional[int] = None
  extend_seq_lens: Optional[torch.Tensor] = None
  extend_prefix_lens: Optional[torch.Tensor] = None
  # extend_start_loc: Optional[torch.Tensor] = None
  extend_prefix_lens_cpu: Optional[List[int]] = None
  extend_seq_lens_cpu: Optional[List[int]] = None

  @classmethod
  def _ensure_pinned_buffers(cls, max_bs: int):
    """确保 pinned memory buffers 已初始化（SGLang 风格优化）"""
    if cls._pinned_temperatures is None or max_bs > cls._max_buffer_size:
      cls._max_buffer_size = max(max_bs, cls._max_buffer_size)
      cls._pinned_temperatures = torch.empty(
        cls._max_buffer_size, dtype=torch.float32, pin_memory=True
      )
      cls._pinned_top_ps = torch.empty(cls._max_buffer_size, dtype=torch.float32, pin_memory=True)
      cls._pinned_top_ks = torch.empty(cls._max_buffer_size, dtype=torch.int64, pin_memory=True)
      cls._pinned_extend_lens = torch.empty(
        cls._max_buffer_size, dtype=torch.int64, pin_memory=True
      )
      cls._pinned_extend_prefix_lens = torch.empty(
        cls._max_buffer_size, dtype=torch.int64, pin_memory=True
      )
      cls._pinned_last_token_indices = torch.empty(
        cls._max_buffer_size, dtype=torch.int64, pin_memory=True
      )

  @classmethod
  def init_new(cls, batch: ScheduledBatch, attn_backend: Optional[AttentionBackend] = None):
    # Compute max_seq_len from CPU tensor to avoid GPU sync
    # 优化：直接从 CPU tensor 取值，避免 .item() 的潜在同步
    max_seq_len = None
    if batch.seq_lens_cpu is not None:
      max_seq_len = int(batch.seq_lens_cpu.max())

    # Pre-convert seq_lens to int32 to avoid repeated conversions in FA backends
    # 优化：使用 non_blocking=True 避免同步
    seq_lens_int32 = (
      batch.seq_lens.to(torch.int32, non_blocking=True) if batch.seq_lens is not None else None
    )

    forward_batch = cls(
      forward_mode=batch.forward_mode,
      batch_size=len(batch.reqs),
      input_ids=batch.input_ids,
      req_pool_indices=batch.req_pool_indices,
      out_cache_loc=batch.out_cache_loc,
      seq_lens=batch.seq_lens,
      seq_lens_int32=seq_lens_int32,
      seq_lens_cpu=batch.seq_lens_cpu,
      max_seq_len=max_seq_len,
      attn_backend=attn_backend,
      all_seqs=batch.reqs,
    )

    # Pre-build per-sequence sampling params on the target device.
    # This avoids repeatedly converting Python lists to tensors inside ModelRunner.sample().
    # 优化：使用预分配的 pinned memory buffer 进行零拷贝异步传输（SGLang 风格）
    if batch.reqs:
      dev = batch.device if batch.device is not None else batch.input_ids.device
      bs = len(batch.reqs)

      # 确保 pinned buffer 已初始化
      cls._ensure_pinned_buffers(bs)

      # Compute max_top_k on CPU to avoid `.item()` sync in batched top-k.
      # Ensure it's always >= 1 so top-k path can safely run even when empty.
      forward_batch.sampling_max_top_k = max(
        1, max((int(r.sampling_params.top_k) for r in batch.reqs), default=0)
      )

      # 提取 sampling params 到 pinned buffer（在 CPU 上操作，无 GPU sync）
      temps = [r.sampling_params.temperature for r in batch.reqs]
      top_ps = [r.sampling_params.top_p for r in batch.reqs]
      top_ks = [r.sampling_params.top_k for r in batch.reqs]

      cls._pinned_temperatures[:bs] = torch.as_tensor(temps, dtype=torch.float32)
      cls._pinned_top_ps[:bs] = torch.as_tensor(top_ps, dtype=torch.float32)
      cls._pinned_top_ks[:bs] = torch.as_tensor(top_ks, dtype=torch.int64)

      # 异步传输到 GPU（真正的零拷贝，无临时 buffer）
      forward_batch.sampling_temperatures = cls._pinned_temperatures[:bs].to(dev, non_blocking=True)
      forward_batch.sampling_top_ps = cls._pinned_top_ps[:bs].to(dev, non_blocking=True)
      forward_batch.sampling_top_ks = cls._pinned_top_ks[:bs].to(dev, non_blocking=True)

    if batch.forward_mode.is_extend():
      # forward_batch.extend_num_tokens = batch.input_ids.shape[0]
      extend_lens_cpu = batch.extend_lens or []
      total_extend_tokens = int(sum(extend_lens_cpu)) if extend_lens_cpu else 0

      dev = batch.device if batch.device is not None else batch.input_ids.device
      bs = len(batch.reqs)
      cls._ensure_pinned_buffers(bs)

      # H2D extend lens/prefix lens via pinned buffers on the current CUDA stream
      cls._pinned_extend_lens[:bs] = torch.as_tensor(extend_lens_cpu, dtype=torch.int64)
      cls._pinned_extend_prefix_lens[:bs] = torch.as_tensor(
        batch.prefix_lens or [], dtype=torch.int64
      )
      forward_batch.extend_seq_lens = cls._pinned_extend_lens[:bs].to(dev, non_blocking=True)
      forward_batch.extend_prefix_lens = cls._pinned_extend_prefix_lens[:bs].to(
        dev, non_blocking=True
      )

      # Precompute last token indices for sampling from [total_tokens, vocab] logits
      # without per-step GPU tensor allocations.
      if total_extend_tokens > 0:
        cumsum = torch.cumsum(cls._pinned_extend_lens[:bs], dim=0, dtype=torch.int64)
        cls._pinned_last_token_indices[:bs] = torch.clamp(cumsum - 1, min=0)
        forward_batch.last_token_indices = cls._pinned_last_token_indices[:bs].to(
          dev, non_blocking=True
        )
      forward_batch.extend_prefix_lens_cpu = batch.prefix_lens
      forward_batch.extend_seq_lens_cpu = extend_lens_cpu
      # positions and start loc for extend
      forward_batch.positions = compute_positions_extend(
        extend_prefix_lens=forward_batch.extend_prefix_lens,
        extend_seq_lens=forward_batch.extend_seq_lens,
        total_tokens=total_extend_tokens,
      )
    else:
      forward_batch.positions = clamp_position(batch.seq_lens)

    return forward_batch

  def debug_metadata(self):
    print("ForwardBatch metadata:")
    print(f"  forward_mode: {self.forward_mode}")
    print(f"  attn_backend type: {self.attn_backend.type() if self.attn_backend else None}")
    print(f"  batch_size: {self.batch_size}")
    print(f"  input_ids: {self.input_ids}")
    print(f"  req_pool_indices: {self.req_pool_indices}")
    print(f"  out_cache_loc: {self.out_cache_loc}")
    print(f"  seq_lens: {self.seq_lens}")
    print(f"  extend_seq_lens: {self.extend_seq_lens}")
    print(f"  extend_prefix_lens: {self.extend_prefix_lens}")


@dataclass
class BatchResult:
  logits: torch.Tensor
  next_token_ids: torch.Tensor


def compute_positions_extend(
  extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, total_tokens: int
) -> torch.Tensor:
  """
  Vectorized position builder for EXTEND/MIXED forward.

  For each sequence i:
    positions_i = [prefix_len_i, prefix_len_i+1, ..., prefix_len_i+extend_len_i-1]
  Then concatenate across sequences.
  """
  if total_tokens <= 0:
    return torch.empty((0,), dtype=torch.int64, device=extend_prefix_lens.device)

  # Exclusive start offsets in the concatenated extend token array.
  start_offsets = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens  # [bs]

  prefix_rep = torch.repeat_interleave(extend_prefix_lens, extend_seq_lens)  # [total]
  start_rep = torch.repeat_interleave(start_offsets, extend_seq_lens)  # [total]
  token_idx = torch.arange(total_tokens, device=extend_prefix_lens.device, dtype=torch.int64)
  within = token_idx - start_rep
  return (prefix_rep + within).to(torch.int64)


# Temporarily disable torch.compile to debug batch issues
# @torch.compile(dynamic=True)
def clamp_position(seq_lens):
  return torch.clamp((seq_lens - 1), min=0).to(torch.int64)
