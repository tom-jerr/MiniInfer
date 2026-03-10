"""KV Cache Manager - 由 Scheduler 持有"""

import triton.language as tl
import triton
from typing import Optional, Tuple
import torch
import logging

from .interface import (
  IKVCacheStorage,
  ITokenAllocator,
  IRequestPool,
  IPrefixCache,
)
from .memory_pool import MHAKVCacheStorage, PagedTokenAllocator, RequestPool
from .radix_cache import RadixCache
from miniinfer.scheduler.scheduler_batch import ScheduledBatch, Req, ForwardMode, ForwardBatch

logger = logging.getLogger(__name__)


class KVCacheManager:
  """
  KV Cache 管理器

  由多个组件构成:
  - CacheStorage: 物理存储 (MHAKVCacheStorage)
  - TokenAllocator: 分配 KV indices (PagedTokenAllocator)
  - RequestPool: 管理不同请求分配
  - PrefixCache: 前缀缓存 (RadixCache)
  - MemoryBudgetManager: 显存预算管理

  初始化流程:
  1. 使用 gpu_memory_utilization 计算 GPU 空闲显存能容纳的 max_total_tokens
  2. 根据 max_total_tokens 初始化物理存储和各组件
  """

  def __init__(
    self,
    size: int,
    max_requests: int = 256,
    max_context_len: int = 4096,
    num_layers: int = 32,
    num_heads: int = 8,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    enable_prefix_cache: bool = True,
    page_size: int = 256,
    max_extend_tokens: int = 8192,
  ):
    """
    初始化 KV Cache 管理器

    Args:
        size: KV cache 容量（token 数），None 时自动根据 GPU 显存计算
        max_requests: 最大并发请求数
        max_context_len: 单请求最大上下文长度
        num_layers: 模型层数
        num_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型
        device: GPU 设备
        enable_prefix_cache: 是否启用前缀缓存
        page_size: 每页 token 数
        max_extend_tokens: 单次 prefill 最大 token 数
    """
    self.max_requests = max_requests
    self.max_context_len = max_context_len
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.dtype = dtype
    self.device = device
    self.enable_prefix_cache = enable_prefix_cache
    self.page_size = page_size
    self.max_extend_tokens = max_extend_tokens

    # for debug
    self.size = size

    # ============ Pinned Memory Buffer Pool (SGLang 风格优化) ============
    # 预分配 pinned memory buffers 用于异步 CPU→GPU 传输
    # 这避免了每次 step 都动态分配 tensor 和临时 pinned buffer 的开销
    logger.info(
      f"Initializing pinned memory buffer pool (max_requests={max_requests}, max_extend_tokens={max_extend_tokens})..."
    )

    # 序列长度相关 (每个 batch 最多 max_requests 个请求)
    self.pinned_seq_lens = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    self.pinned_prefix_lens = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    self.pinned_extend_lens = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    self.pinned_req_pool_indices = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)
    # Decode input ids (one token per request)
    self.pinned_decode_input_ids = torch.empty(max_requests, dtype=torch.int64, pin_memory=True)

    # Token IDs (extend 阶段可能有很多 token)
    self.pinned_input_ids = torch.empty(max_extend_tokens, dtype=torch.int64, pin_memory=True)

    logger.info("Pinned memory buffer pool initialized successfully")

    # 初始化物理存储
    self.storage: IKVCacheStorage = MHAKVCacheStorage(
      size=size,
      page_size=page_size,
      num_layers=num_layers,
      num_heads=num_heads,
      head_dim=head_dim,
      dtype=dtype,
      device=device,
    )

    # 初始化 Token 分配器
    self.token_allocator: ITokenAllocator = PagedTokenAllocator(
      size=size,
      device=device,
      page_size=page_size,
    )

    # 初始化请求池
    self.request_pool: IRequestPool = RequestPool(
      max_requests=max_requests,
      max_context_len=max_context_len,
      device=device,
    )

    # 初始化前缀缓存
    if enable_prefix_cache:
      self.prefix_cache: IPrefixCache = RadixCache(
        token_allocator=self.token_allocator,
        page_size=page_size,
      )
    else:
      self.prefix_cache = None

    # prefill max length
    self.max_extend_tokens = max_extend_tokens

    logger.info(
      f"Initialized KVCacheManager with size={size} tokens, "
      f"max_requests={max_requests}, max_context_len={max_context_len}, "
      f"num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}, "
      f"dtype={dtype}, device={device}, enable_prefix_cache={enable_prefix_cache}, "
      f"page_size={page_size}, max_extend_tokens={max_extend_tokens}"
    )

  # ============== public methods for scheduler ==============

  def warmup_release_path(self) -> None:
    """
    Warm up CUDA ops used by the deferred release path.

    The first real `drain_pending_releases()` can otherwise pay for lazy module
    loading (e.g. request-pool read cast and grouped free/unique kernels). Use
    a no-op warmup that leaves allocator/request-pool state unchanged.
    """
    if not torch.cuda.is_available() or str(self.device) == "cpu":
      return

    # Warm request-pool read/cast used to fetch kv_indices during release.
    warmup_tokens = max(1, min(int(self.page_size), int(self.max_context_len)))
    _ = self.request_pool.read(0, slice(0, warmup_tokens))

    free_pages = getattr(self.token_allocator, "free_pages", None)
    if free_pages is None or free_pages.numel() == 0:
      torch.cuda.synchronize()
      return

    # Warm grouped free/flush kernels using an already-free page so allocator
    # contents stay logically unchanged after dedup.
    dummy_page = free_pages[:1].clone()
    page_offsets = torch.arange(self.page_size, dtype=torch.int64, device=self.device)
    dummy_free = dummy_page * self.page_size + page_offsets

    begin_group = getattr(self.token_allocator, "begin_free_group", None)
    end_group = getattr(self.token_allocator, "end_free_group", None)
    if begin_group is not None and end_group is not None:
      begin_group()
      try:
        self.token_allocator.free(dummy_free)
      finally:
        end_group()
    else:
      self.token_allocator.free(dummy_free)

    torch.cuda.synchronize()

  def prefix_for_waiting_req(self, req: "Req"):
    """Req is mutable"""
    req.fill_ids = req.origin_input_ids + req.output_ids
    max_prefix_len = max(len(req.fill_ids) - 1, 0)
    token_ids = req.fill_ids[:max_prefix_len]
    if self.prefix_cache is not None:
      kv_indices, node = self.prefix_cache.match_prefix(token_ids)
      req.prefix_indices = kv_indices
      req.cache_protected_len = kv_indices.size(0)
      req.last_node = node
      # Lock matched prefix so it won't be evicted while this request is using it.
      # NOTE: match_prefix may return the root node when prefix_len==0; locking is a no-op then.
      self.prefix_cache.inc_lock_ref(node)
    else:
      # When prefix cache is disabled, still expose an empty prefix tensor for downstream logic.
      req.prefix_indices = torch.empty((0,), dtype=torch.int64, device=self.device)
      req.cache_protected_len = 0
      req.last_node = None
    req.extend_input_len = len(req.fill_ids) - req.cache_protected_len
    logger.debug(
      f"prefix_for_waiting_req: req_id={req.req_id}, prefix_len={req.cache_protected_len}, extend_input_len={req.extend_input_len}"
    )

  def prepare_for_extend(
    self,
    batch: "ScheduledBatch",
  ):
    """batch is mutable"""

    # Init batch metadata
    batch.forward_mode = ForwardMode.EXTEND
    extend_ids = [r.fill_ids[len(r.prefix_indices) :] for r in batch.reqs]
    extend_num_tokens = sum(len(ids) for ids in extend_ids)
    seq_lens = [len(r.fill_ids) for r in batch.reqs]
    prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
    extend_lens = [r.extend_input_len for r in batch.reqs]
    bs = len(batch.reqs)

    # 优化：使用预分配的 pinned memory buffer 进行零拷贝异步传输
    # seq_lens
    self.pinned_seq_lens[:bs] = torch.as_tensor(seq_lens, dtype=torch.int64)
    seq_lens_tensor = self.pinned_seq_lens[:bs].to(self.device, non_blocking=True)
    seq_lens_cpu = self.pinned_seq_lens[:bs].clone()  # CPU 副本

    # extend_ids (flatten)
    extend_ids_flat = [token_id for ids in extend_ids for token_id in ids]
    num_tokens = len(extend_ids_flat)
    self.pinned_input_ids[:num_tokens] = torch.as_tensor(extend_ids_flat, dtype=torch.int64)
    extend_ids_tensor = self.pinned_input_ids[:num_tokens].to(self.device, non_blocking=True)

    batch.prefix_lens = prefix_lens
    batch.extend_lens = extend_lens
    batch.seq_lens = seq_lens_tensor
    batch.seq_lens_cpu = seq_lens_cpu
    batch.input_ids = extend_ids_tensor

    # KV cache related metadata
    bs = len(batch.reqs)
    prefix_tensors = [r.prefix_indices for r in batch.reqs]
    # Allocate req slots
    req_pool_indices = self.request_pool.alloc(bs)
    if req_pool_indices is None:
      raise RuntimeError(
        f"Failed to allocate request pool slots: "
        f"requested {bs} slots, but request pool is full. "
        f"This usually indicates a scheduling bug where batch size exceeds max_num_seqs."
      )
    # Persist request-pool indices on the Req objects for correct release/update flows.
    for req, pool_idx in zip(batch.reqs, req_pool_indices):
      req.req_pool_idx = int(pool_idx)
    # 优化：使用预分配的 pinned memory buffer
    self.pinned_req_pool_indices[:bs] = torch.as_tensor(req_pool_indices, dtype=torch.int64)
    req_pool_indices_tensor = self.pinned_req_pool_indices[:bs].to(self.device, non_blocking=True)

    # Allocate KV cache slots
    # 检查物理空闲空间，而非 available_tokens()（包含 evictable）
    if self.token_allocator.available_size() < extend_num_tokens and self.prefix_cache is not None:
      needed_to_evict = extend_num_tokens - self.token_allocator.available_size()
      needed_to_evict = (
        (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      )  # 向上对齐到 page_size
      logger.debug(
        "prepare_for_extend: need to evict for extend batch, "
        f"extend_num_tokens={extend_num_tokens}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    # 优化：为避免 CUDA 同步，先在 CPU 上创建负值占位符，然后一次性 concat
    last_loc_cpu = [
      (t[-1:] if len(t) > 0 else torch.tensor([-1], dtype=torch.int64)) for t in prefix_tensors
    ]
    # torch.cat 需要 list/tuple，产生 shape (bs,)
    last_loc_concat = torch.cat(last_loc_cpu)
    # 如果 prefix_tensors 不在 GPU 上，异步传输
    if last_loc_concat.device != self.device:
      last_loc = last_loc_concat.to(self.device, non_blocking=True)
    else:
      last_loc = last_loc_concat

    # 优化：使用预分配的 pinned memory buffer
    self.pinned_prefix_lens[:bs] = torch.as_tensor(batch.prefix_lens, dtype=torch.int64)
    prefix_lens_device = self.pinned_prefix_lens[:bs].to(self.device, non_blocking=True)
    prefix_lens_cpu = self.pinned_prefix_lens[:bs].clone()

    out_cache_loc = self.token_allocator.alloc_pages_extend(
      prefix_lens=prefix_lens_device,
      prefix_lens_cpu=prefix_lens_cpu,
      seq_lens=batch.seq_lens,
      seq_lens_cpu=batch.seq_lens_cpu,
      last_loc=last_loc,
      extend_num_tokens=extend_num_tokens,
    )

    # 检查分配是否成功
    if out_cache_loc is None:
      physical_available = self.token_allocator.available_size()
      logical_available = self.available_tokens()
      raise RuntimeError(
        f"Failed to allocate KV cache for extend batch: "
        f"need {extend_num_tokens} tokens ({bs} requests), "
        f"physical_available {physical_available} tokens (free pages), "
        f"logical_available {logical_available} tokens (includes evictable radix)"
      )

    # Write prefix cache and new extend cache to request pool
    # 优化：使用预分配的 pinned memory buffer
    self.pinned_extend_lens[:bs] = torch.as_tensor(extend_lens, dtype=torch.int64)
    extend_lens_device = self.pinned_extend_lens[:bs].to(self.device, non_blocking=True)

    write_cache_indices(
      out_cache_loc,
      req_pool_indices_tensor,
      prefix_lens_device,
      batch.seq_lens,
      extend_lens_device,
      prefix_tensors,
      self.request_pool,
    )

    batch.req_pool_indices = req_pool_indices_tensor
    batch.out_cache_loc = out_cache_loc

  def prepare_for_mixed(
    self,
    batch: "ScheduledBatch",
    extend_reqs: list,
    decode_reqs: list,
    *,
    decode_input_ids: Optional[torch.Tensor] = None,
    decode_seq_lens: Optional[torch.Tensor] = None,
    decode_seq_lens_cpu: Optional[torch.Tensor] = None,
  ):
    """
    为 mixed batch (prefill + decode) 准备 KV cache。

    Mixed batch 将 decode 请求视为 extend_len=1 的 extend 请求，
    这样所有请求统一使用 extend kernel 处理。

    Args:
        batch: 包含所有请求 (extend_reqs + decode_reqs) 的 ScheduledBatch
        extend_reqs: prefill/chunked prefill extend 请求列表
        decode_reqs: 正在 decode 的请求列表（已有 req_pool_idx 和 KV cache）
        decode_input_ids: (overlap 路径) 预设的 decode input_ids（含 -future_indices 占位符），
            不为 None 时跳过 r.output_ids[-1] 读取
        decode_seq_lens: (overlap 路径) decode 请求的 seq_lens（GPU tensor），
            不为 None 时跳过从 len(r.output_ids) 计算
        decode_seq_lens_cpu: (overlap 路径) decode 请求的 seq_lens（CPU tensor）
    """
    if not decode_reqs:
      # 没有 decode 请求，退化为纯 extend
      self.prepare_for_extend(batch)
      return

    # ================== CPU-side metadata build (no CUDA sync) ==================
    extend_bs = len(extend_reqs)
    decode_bs = len(decode_reqs)
    total_bs = extend_bs + decode_bs

    # Extend requests (prefill tokens)
    extend_seq_lens: list[int] = []
    extend_prefix_lens: list[int] = []
    extend_extend_lens: list[int] = []
    extend_prefix_tensors: list[torch.Tensor] = []
    extend_ids_flat: list[int] = []
    extend_num_tokens = 0
    if extend_bs > 0:
      for r in extend_reqs:
        prefix_len = len(r.prefix_indices)
        ids = r.fill_ids[prefix_len:]
        extend_ids_flat.extend(ids)
        extend_num_tokens += len(ids)
        extend_seq_lens.append(len(r.fill_ids))
        extend_prefix_lens.append(prefix_len)
        extend_extend_lens.append(int(r.extend_input_len))
        extend_prefix_tensors.append(r.prefix_indices)

    # Decode requests: we treat each decode as extend_len=1.
    total_new_tokens = extend_num_tokens + decode_bs

    # Decode input ids: either placeholder (overlap) or last generated tokens.
    if decode_input_ids is not None:
      decode_input_ids_dev = decode_input_ids.to(self.device, non_blocking=True)
    else:
      # Synchronous/token-based path: gather last tokens on CPU and H2D via pinned buffer.
      for i, r in enumerate(decode_reqs):
        last_token_id = r.get_last_token()
        if last_token_id is None:
          last_token_id = r.output_ids[-1]
        self.pinned_decode_input_ids[i] = int(last_token_id)
      decode_input_ids_dev = self.pinned_decode_input_ids[:decode_bs].to(
        self.device, non_blocking=True
      )

    # Decode seq lens metadata must be CPU-known for control flow and to avoid `.item()` sync.
    # Semantics: these are the *current* sequence lengths BEFORE allocating this step's decode token.
    # This function will increment them in-place after allocating the decode KV slot.
    if decode_seq_lens_cpu is None:
      # Fallback: build from Req.current_seq_len (CPU-known).
      tmp = torch.empty(decode_bs, dtype=torch.int64)
      for i, r in enumerate(decode_reqs):
        tmp[i] = int(r.current_seq_len)
      decode_seq_lens_cpu = tmp
    else:
      decode_seq_lens_cpu = decode_seq_lens_cpu[:decode_bs]

    if decode_seq_lens is None:
      # Fallback: build a device tensor from CPU lengths via pinned buffer (async H2D).
      self.pinned_seq_lens[:decode_bs].copy_(decode_seq_lens_cpu)
      decode_seq_lens = self.pinned_seq_lens[:decode_bs].to(self.device, non_blocking=True)
    else:
      decode_seq_lens = decode_seq_lens[:decode_bs]

    # ================== KV allocations (runs on current CUDA stream) ==================
    # Evict if needed - check physical free pages
    if self.token_allocator.available_size() < total_new_tokens and self.prefix_cache is not None:
      needed_to_evict = total_new_tokens - self.token_allocator.available_size()
      needed_to_evict = (
        (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      )
      logger.debug(
        "prepare_for_mixed: need to evict, "
        f"total_new_tokens={total_new_tokens}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    # ---- Decode reqs: allocate 1 token KV slot each, then increment seq_lens in-place ----
    decode_req_pool_indices = [int(r.req_pool_idx) for r in decode_reqs]
    if any(idx < 0 for idx in decode_req_pool_indices):
      raise RuntimeError("prepare_for_mixed: decode_reqs contains invalid req_pool_idx (<0).")

    # H2D req_pool_indices via pinned buffer
    self.pinned_req_pool_indices[:decode_bs] = torch.as_tensor(
      decode_req_pool_indices, dtype=torch.int64
    )
    decode_rpi = self.pinned_req_pool_indices[:decode_bs].to(self.device, non_blocking=True)

    # last_loc for each decode req: position = (current_seq_len - 1)
    last_loc = self.request_pool.req_to_token_pool()[decode_rpi, decode_seq_lens - 1]
    decode_out_cache_loc = self.token_allocator.alloc_pages_decode(
      seq_lens=decode_seq_lens,
      seq_lens_cpu=decode_seq_lens_cpu,
      last_loc=last_loc,
    )
    if decode_out_cache_loc is None:
      physical_available = self.token_allocator.available_size()
      logical_available = self.available_tokens()
      raise RuntimeError(
        "Failed to allocate KV cache for mixed batch decode part: "
        f"need {decode_bs} tokens ({decode_bs} decode reqs), "
        f"physical_available {physical_available} tokens (free pages), "
        f"logical_available {logical_available} tokens (includes evictable radix)"
      )

    # Update request->kv_loc mapping for the newly allocated decode token:
    # new_pos = current_seq_len (0-indexed position of the new token)
    new_pos = decode_seq_lens
    self.request_pool.req_to_token_pool()[decode_rpi, new_pos] = decode_out_cache_loc.to(torch.int32)

    # Increment decode seq_lens for forward (now includes the newly allocated token)
    decode_seq_lens.add_(1)
    decode_seq_lens_cpu.add_(1)

    # After increment: prefix_len = seq_len - 1, extend_len = 1
    decode_prefix_lens = (decode_seq_lens_cpu - 1).tolist()
    decode_extend_lens = [1] * decode_bs

    # ---- Extend reqs: allocate req_pool slot + KV cache for all prefill tokens ----
    extend_req_pool_indices: list[int] = []
    if extend_bs > 0:
      pool_indices = self.request_pool.alloc(extend_bs)
      if pool_indices is None:
        raise RuntimeError(
          f"Failed to allocate request pool slots for extend requests: requested {extend_bs} slots."
        )
      for req, pool_idx in zip(extend_reqs, pool_indices):
        req.req_pool_idx = int(pool_idx)
      extend_req_pool_indices = list(pool_indices)

      # Prepare extend input_ids via pinned buffer (async H2D)
      if extend_num_tokens > 0:
        self.pinned_input_ids[:extend_num_tokens] = torch.as_tensor(extend_ids_flat, dtype=torch.int64)
        extend_ids_tensor = self.pinned_input_ids[:extend_num_tokens].to(
          self.device, non_blocking=True
        )
      else:
        extend_ids_tensor = torch.empty((0,), dtype=torch.int64, device=self.device)

      # Prepare seq_lens/prefix_lens/extend_lens via pinned buffers (async H2D)
      self.pinned_seq_lens[:extend_bs] = torch.as_tensor(extend_seq_lens, dtype=torch.int64)
      extend_seq_lens_device = self.pinned_seq_lens[:extend_bs].to(self.device, non_blocking=True)
      extend_seq_lens_cpu = self.pinned_seq_lens[:extend_bs].clone()

      self.pinned_prefix_lens[:extend_bs] = torch.as_tensor(extend_prefix_lens, dtype=torch.int64)
      extend_prefix_lens_device = self.pinned_prefix_lens[:extend_bs].to(
        self.device, non_blocking=True
      )
      extend_prefix_lens_cpu = self.pinned_prefix_lens[:extend_bs].clone()

      self.pinned_extend_lens[:extend_bs] = torch.as_tensor(extend_extend_lens, dtype=torch.int64)
      extend_lens_device = self.pinned_extend_lens[:extend_bs].to(self.device, non_blocking=True)

      # last_loc for extend alloc: use CPU-built placeholder and async H2D
      last_loc_cpu = [
        (t[-1:] if len(t) > 0 else torch.tensor([-1], dtype=torch.int64))
        for t in extend_prefix_tensors
      ]
      last_loc_concat = torch.cat(last_loc_cpu)
      last_loc = last_loc_concat.to(self.device, non_blocking=True)

      extend_out_cache_loc = self.token_allocator.alloc_pages_extend(
        prefix_lens=extend_prefix_lens_device,
        prefix_lens_cpu=extend_prefix_lens_cpu,
        seq_lens=extend_seq_lens_device,
        seq_lens_cpu=extend_seq_lens_cpu,
        last_loc=last_loc,
        extend_num_tokens=extend_num_tokens,
      )
      if extend_out_cache_loc is None:
        physical_available = self.token_allocator.available_size()
        logical_available = self.available_tokens()
        raise RuntimeError(
          "Failed to allocate KV cache for mixed batch extend part: "
          f"need {extend_num_tokens} tokens, "
          f"physical_available {physical_available} tokens (free pages), "
          f"logical_available {logical_available} tokens (includes evictable radix)"
        )

      # Write prefix + extend indices to request pool
      self.pinned_req_pool_indices[:extend_bs] = torch.as_tensor(
        extend_req_pool_indices, dtype=torch.int64
      )
      extend_rpi = self.pinned_req_pool_indices[:extend_bs].to(self.device, non_blocking=True)

      write_cache_indices(
        extend_out_cache_loc,
        extend_rpi,
        extend_prefix_lens_device,
        extend_seq_lens_device,
        extend_lens_device,
        extend_prefix_tensors,
        self.request_pool,
      )
    else:
      extend_out_cache_loc = torch.empty((0,), dtype=torch.int64, device=self.device)
      extend_ids_tensor = torch.empty((0,), dtype=torch.int64, device=self.device)

    # ================== Batch tensors (forward inputs/metadata) ==================
    # input_ids: [extend_tokens..., decode_tokens...]
    if extend_num_tokens > 0:
      batch.input_ids = torch.empty((total_new_tokens,), dtype=torch.int64, device=self.device)
      batch.input_ids[:extend_num_tokens].copy_(extend_ids_tensor)
      batch.input_ids[extend_num_tokens:].copy_(decode_input_ids_dev)
    else:
      batch.input_ids = decode_input_ids_dev

    # seq_lens: [extend_seq_lens..., decode_seq_lens(after+1)...]
    if extend_bs > 0:
      # extend_seq_lens_device is defined only when extend_bs>0
      batch.seq_lens = torch.empty((total_bs,), dtype=torch.int64, device=self.device)
      batch.seq_lens[:extend_bs].copy_(extend_seq_lens_device)
      batch.seq_lens[extend_bs:].copy_(decode_seq_lens)
      seq_lens_cpu = torch.empty((total_bs,), dtype=torch.int64)
      seq_lens_cpu[:extend_bs] = torch.as_tensor(extend_seq_lens, dtype=torch.int64)
      seq_lens_cpu[extend_bs:] = decode_seq_lens_cpu
      batch.seq_lens_cpu = seq_lens_cpu
    else:
      batch.seq_lens = decode_seq_lens
      batch.seq_lens_cpu = decode_seq_lens_cpu

    batch.prefix_lens = extend_prefix_lens + decode_prefix_lens
    batch.extend_lens = extend_extend_lens + decode_extend_lens

    # req_pool_indices: per-seq mapping
    all_req_pool_indices = extend_req_pool_indices + decode_req_pool_indices
    self.pinned_req_pool_indices[:total_bs] = torch.as_tensor(all_req_pool_indices, dtype=torch.int64)
    batch.req_pool_indices = self.pinned_req_pool_indices[:total_bs].to(self.device, non_blocking=True)

    # out_cache_loc: per-token mapping aligned with input_ids
    if extend_num_tokens > 0:
      batch.out_cache_loc = torch.empty((total_new_tokens,), dtype=torch.int64, device=self.device)
      batch.out_cache_loc[:extend_num_tokens].copy_(extend_out_cache_loc)
      batch.out_cache_loc[extend_num_tokens:].copy_(decode_out_cache_loc)
    else:
      batch.out_cache_loc = decode_out_cache_loc

    return

  def prepare_for_decode(self, batch: "ScheduledBatch", skip_input_ids: bool = False):
    batch.forward_mode = ForwardMode.DECODE
    # Decode 阶段的 input_ids 是每个请求最后生成的 token
    # 从每个 req.output_ids[-1] 获取
    bs = len(batch.reqs)
    if bs == 0:
      return

    if not skip_input_ids:
      # 非 overlap 路径：从 CPU 侧收集 last_token，并通过 pinned buffer 异步 H2D。
      # 这避免了 Python 循环逐元素写 GPU tensor（会产生大量 tiny kernel/allocator 交互）。
      for i, req in enumerate(batch.reqs):
        last_token = req.get_last_token()
        if last_token is None:
          # 回退（理论上不应该发生）
          last_token = req.output_ids[-1]
        self.pinned_decode_input_ids[i] = int(last_token)
      batch.input_ids = self.pinned_decode_input_ids[:bs].to(self.device, non_blocking=True)
    else:
      batch.input_ids = batch.output_ids
    batch.output_ids = None
    token_per_req = 1  # decode

    # Allocate KV cache slots - 检查物理空闲空间
    if self.token_allocator.available_size() < bs * token_per_req and self.prefix_cache is not None:
      needed_to_evict = (bs * token_per_req) - self.token_allocator.available_size()
      needed_to_evict = (
        (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      )  # 向上对齐到 page_size
      logger.debug(
        "prepare_for_decode: need to evict for decode batch, "
        f"num_tokens={bs * token_per_req}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    last_loc = self.request_pool.req_to_token_pool()[batch.req_pool_indices, batch.seq_lens - 1]
    out_cache_loc = self.token_allocator.alloc_pages_decode(
      seq_lens=batch.seq_lens,
      seq_lens_cpu=batch.seq_lens_cpu,  # calc num pages
      last_loc=last_loc,
    )

    # 检查分配是否成功
    if out_cache_loc is None:
      physical_available = self.token_allocator.available_size()
      logical_available = self.available_tokens()
      raise RuntimeError(
        f"Failed to allocate KV cache for decode batch: "
        f"need {bs} tokens ({bs} requests), "
        f"physical_available {physical_available} tokens (free pages), "
        f"logical_available {logical_available} tokens (includes evictable radix)"
      )

    batch.out_cache_loc = out_cache_loc
    batch.seq_lens.add_(1)
    batch.seq_lens_cpu.add_(1)
    # Update request -> kv_loc mapping for the newly allocated decode token.
    # This is needed to build correct block/page tables for flash-attn backends.
    new_pos = batch.seq_lens - 1
    self.request_pool.req_to_token_pool()[batch.req_pool_indices, new_pos] = out_cache_loc.to(
      torch.int32
    )

  def release_request(
    self,
    req: Req,
    is_insert: bool = True,
  ):
    """
    释放请求的 KV cache

    Args:
        req_idx: 请求索引
        token_ids: 完整的 token ids (用于缓存到 radix tree)
        num_tokens: 总 token 数量
        cache_to_radix: 是否缓存到 radix tree
    """
    # 读取所有 KV indices
    req_pool_idx = int(getattr(req, "req_pool_idx", -1))
    if req_pool_idx < 0:
      raise RuntimeError(
        f"release_request called with invalid req_pool_idx={req_pool_idx} "
        f"for req_id={req.req_id}. This indicates request lifecycle metadata corruption."
      )
    token_ids = req.origin_input_ids + req.output_ids
    num_tokens = len(token_ids)
    kv_indices = self.request_pool.read(req_pool_idx, slice(0, num_tokens))

    if is_insert and self.prefix_cache is not None:
      # 计算 page 对齐长度
      page_aligned_len = (num_tokens // self.page_size) * self.page_size

      # 插入到前缀缓存（会截断到 page 对齐）
      # insert 返回匹配的前缀长度（insert 之前已在 cache 中的部分）
      new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

      # 释放逻辑：
      # - kv_indices[0:cache_protected_len] 由请求开始时的 lock 保护，不释放
      # - kv_indices[cache_protected_len:new_prefix_len] insert 前已存在（重复部分），可释放
      # - kv_indices[new_prefix_len:page_aligned_len] 被新创建的节点持有，不应释放
      # - kv_indices[page_aligned_len:num_tokens] 被截断未保存到 radix tree，应释放

      # 释放重复部分（如果有）
      if new_prefix_len > req.cache_protected_len:
        self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])

      # 释放被截断的部分（如果有）
      if page_aligned_len < num_tokens:
        self.token_allocator.free(kv_indices[page_aligned_len:])
    else:
      # 没有 prefix cache，释放所有未保护的部分
      self.token_allocator.free(kv_indices[req.cache_protected_len :])

    self.request_pool.free([req_pool_idx])  # param is list
    if self.prefix_cache is not None:
      self.prefix_cache.dec_lock_ref(req.last_node)

  def update_finished_req_radix_cache(self, req: Req, is_insert: bool = True):
    """更新前缀缓存"""
    if int(getattr(req, "req_pool_idx", -1)) < 0:
      raise RuntimeError(
        f"update_finished_req_radix_cache called with invalid req_pool_idx="
        f"{getattr(req, 'req_pool_idx', -1)} for req_id={req.req_id}"
      )
    token_ids = req.origin_input_ids + req.output_ids
    num_tokens = len(token_ids)
    kv_indices = self.request_pool.read(req.req_pool_idx, slice(0, num_tokens))

    if is_insert and self.prefix_cache is not None:
      # 计算 page 对齐长度
      page_aligned_len = (num_tokens // self.page_size) * self.page_size

      # 插入到前缀缓存
      new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

      # 同 release_request 的释放逻辑
      # 释放重复部分（如果有）
      if new_prefix_len > req.cache_protected_len:
        self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])

      # 释放被截断的部分（如果有）
      if page_aligned_len < num_tokens:
        self.token_allocator.free(kv_indices[page_aligned_len:])
    else:
      # 没有 prefix cache，释放所有未保护的部分
      self.token_allocator.free(kv_indices[req.cache_protected_len :])

    self.request_pool.free([req.req_pool_idx])  # param is list
    if self.prefix_cache is not None:
      self.prefix_cache.dec_lock_ref(req.last_node)

  def update_unfinished_req_radix_cache(self, req: Req):
    """更新前缀缓存"""
    if self.prefix_cache is None:
      return
    token_ids = req.fill_ids
    kv_indices = self.request_pool.read(req.req_pool_idx, slice(0, len(token_ids)))
    # new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

    # BUG FIX 注意：这里不释放内存！
    # 原因：kv_indices 中 [cache_protected_len:new_prefix_len] 部分是重复的，
    # 但 [new_prefix_len:] 部分可能被新节点持有。
    # 如果在这里释放，后续 update_finished_req_radix_cache 或 release_request
    # 会再次尝试释放相同的页面，导致 double free。
    # 正确的做法是在 release_request 统一处理释放逻辑。

    # update req metadata
    new_indices, new_last_node = self.prefix_cache.match_prefix(token_ids)
    self.request_pool.write(
      req.req_pool_idx,
      slice(req.cache_protected_len, len(new_indices)),
      new_indices[req.cache_protected_len :],
    )

    self.prefix_cache.dec_lock_ref(req.last_node)
    self.prefix_cache.inc_lock_ref(new_last_node)

    req.cache_protected_len = len(new_indices)
    req.prefix_indices = new_indices
    req.last_node = new_last_node

  # =============== Attention forward calling ================
  def get_page_table(self, forward_batch: ForwardBatch, seq_len: torch.Tensor) -> torch.Tensor:
    """获取指定 batch 的 page table"""
    return self.request_pool.req_to_token_pool()[forward_batch.req_pool_indices, :seq_len]

  def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """获取指定层的 KV buffer"""
    return self.storage.get_kv_buffer(layer_id)

  def set_kv_buffer(
    self,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
    cache_v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
  ):
    """写入 KV cache"""

    self.storage.set_kv_buffer(layer_id, loc, cache_k, cache_v)

  def available_tokens(self) -> int:
    """返回可用的 token 槽位数"""
    available = (
      self.token_allocator.available_size() + self.prefix_cache.evictable_size()
      if self.prefix_cache is not None
      else self.token_allocator.available_size()
    )
    # Logical availability can never exceed total KV capacity.
    if available > self.size:
      logger.warning(
        "available_tokens overflow detected: available=%d > size=%d; clamping",
        available,
        self.size,
      )
      return self.size
    return available

  def can_allocate(self, num_tokens: int) -> bool:
    """检查是否可以分配指定数量的 tokens"""
    return self.available_tokens() >= num_tokens

  def get_stats(self) -> dict:
    """获取统计信息"""
    return {
      "size": self.size,
      "available_tokens": self.available_tokens(),
      "used_tokens": self.size - self.available_tokens(),
      "utilization": 1.0 - self.available_tokens() / self.size,
      "kv_cache_bytes": self.storage.get_kv_size_bytes(),
    }


@triton.jit
def write_req_to_token_pool_triton(
  req_to_token_ptr,  # [max_batch, max_context_len]
  req_pool_indices,
  prefix_tensors,
  pre_lens,
  seq_lens,
  extend_lens,
  out_cache_loc,
  req_to_token_ptr_stride: tl.constexpr,
):
  BLOCK_SIZE: tl.constexpr = 512
  pid = tl.program_id(0)

  req_pool_index = tl.load(req_pool_indices + pid)
  pre_len = tl.load(pre_lens + pid)
  seq_len = tl.load(seq_lens + pid)
  prefix_tensor = tl.load(prefix_tensors + pid).to(tl.pointer_type(tl.int64))

  # write prefix
  num_loop = tl.cdiv(pre_len, BLOCK_SIZE)
  for i in range(num_loop):
    offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
    mask = offset < pre_len
    value = tl.load(prefix_tensor + offset, mask=mask)
    tl.store(
      req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset,
      value,
      mask=mask,
    )

  # NOTE: This can be slow for large bs
  cumsum_start = tl.cast(0, tl.int64)
  for i in range(pid):
    cumsum_start += tl.load(extend_lens + i)

  num_loop = tl.cdiv(seq_len - pre_len, BLOCK_SIZE)
  for i in range(num_loop):
    offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
    mask = offset < (seq_len - pre_len)
    value = tl.load(out_cache_loc + cumsum_start + offset, mask=mask)
    tl.store(
      req_to_token_ptr + req_pool_index * req_to_token_ptr_stride + offset + pre_len,
      value,
      mask=mask,
    )


def write_cache_indices(
  out_cache_loc: torch.Tensor,
  req_pool_indices_tensor: torch.Tensor,
  prefix_lens_tensor: torch.Tensor,
  seq_lens_tensor: torch.Tensor,
  extend_lens_tensor: torch.Tensor,
  prefix_tensors: list[torch.Tensor],
  req_to_token_pool: RequestPool,
):
  prefix_pointers = torch.tensor(
    [t.data_ptr() for t in prefix_tensors],
    device=req_to_token_pool.device,
    dtype=torch.uint64,
  )
  # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)
  write_req_to_token_pool_triton[(req_pool_indices_tensor.shape[0],)](
    req_to_token_pool.req_to_token_pool(),
    req_pool_indices_tensor,
    prefix_pointers,
    prefix_lens_tensor,
    seq_lens_tensor,
    extend_lens_tensor,
    out_cache_loc,
    req_to_token_pool.req_to_token_pool().shape[1],
  )
