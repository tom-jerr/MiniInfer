"""KV Cache Manager - 由 Scheduler 持有"""

import triton.language as tl
import triton
from typing import Optional, List, Tuple, Any
import torch
import logging

from kvcache.interface import (
  IKVCacheStorage,
  ITokenAllocator,
  IRequestPool,
  IPrefixCache,
)
from kvcache.memory_pool import MHAKVCacheStorage, PagedTokenAllocator, RequestPool
from kvcache.radix_cache import RadixCache
from scheduler.scheduler_batch import ScheduledBatch, Req, ForwardMode, ForwardBatch

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

    extend_ids_tensor = torch.tensor(
      [token_id for ids in extend_ids for token_id in ids], dtype=torch.int64
    ).to(self.device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(self.device)
    seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

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
    req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(self.device)

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

    last_loc = [
      (t[-1:] if len(t) > 0 else torch.tensor([-1], device=self.device)) for t in prefix_tensors
    ]
    # torch.cat produces shape (bs,) which kernel expects; torch.stack would give (bs, 1)
    out_cache_loc = self.token_allocator.alloc_pages_extend(
      prefix_lens=torch.tensor(batch.prefix_lens, dtype=torch.int64).to(self.device),
      prefix_lens_cpu=torch.tensor(batch.prefix_lens, dtype=torch.int64),
      seq_lens=batch.seq_lens,
      seq_lens_cpu=batch.seq_lens_cpu,
      last_loc=torch.cat(last_loc),
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
    prefix_lens_device = torch.tensor(prefix_lens, dtype=torch.int64).to(self.device)
    extend_lens_device = torch.tensor(extend_lens, dtype=torch.int64).to(self.device)
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
  ):
    """
    为 mixed batch (prefill + decode) 准备 KV cache。

    Mixed batch 将 decode 请求视为 extend_len=1 的 extend 请求，
    这样所有请求统一使用 extend kernel 处理。

    Args:
        batch: 包含所有请求 (extend_reqs + decode_reqs) 的 ScheduledBatch
        extend_reqs: prefill/chunked prefill extend 请求列表
        decode_reqs: 正在 decode 的请求列表（已有 req_pool_idx 和 KV cache）
    """
    if not decode_reqs:
      # 没有 decode 请求，退化为纯 extend
      self.prepare_for_extend(batch)
      return

    # ============ 处理 extend 请求 (新 prefill) ============
    extend_ids_list = []
    extend_seq_lens = []
    extend_prefix_lens = []
    extend_extend_lens = []
    extend_prefix_tensors = []

    for r in extend_reqs:
      ids = r.fill_ids[len(r.prefix_indices) :]
      extend_ids_list.append(ids)
      extend_seq_lens.append(len(r.fill_ids))
      extend_prefix_lens.append(len(r.prefix_indices))
      extend_extend_lens.append(r.extend_input_len)
      extend_prefix_tensors.append(r.prefix_indices)

    extend_num_tokens = sum(len(ids) for ids in extend_ids_list)

    # ============ 处理 decode 请求 (视为 extend_len=1) ============
    decode_ids_list = []
    decode_seq_lens = []
    decode_prefix_lens = []
    decode_extend_lens = []

    for r in decode_reqs:
      # decode 请求的 "extend" 就是最后一个 output token
      last_token_id = r.output_ids[-1]
      decode_ids_list.append([last_token_id])
      cur_seq_len = len(r.origin_input_ids) + len(r.output_ids)
      decode_seq_lens.append(cur_seq_len)
      decode_prefix_lens.append(cur_seq_len - 1)  # 前面全是 "prefix"
      decode_extend_lens.append(1)

    decode_num_tokens = len(decode_reqs)
    total_new_tokens = extend_num_tokens + decode_num_tokens

    # ============ 合并 batch metadata ============
    all_ids = []
    for ids in extend_ids_list:
      all_ids.extend(ids)
    for ids in decode_ids_list:
      all_ids.extend(ids)

    all_seq_lens = extend_seq_lens + decode_seq_lens
    all_prefix_lens = extend_prefix_lens + decode_prefix_lens
    all_extend_lens = extend_extend_lens + decode_extend_lens

    batch.input_ids = torch.tensor(all_ids, dtype=torch.int64).to(self.device)
    batch.seq_lens = torch.tensor(all_seq_lens, dtype=torch.int64).to(self.device)
    batch.seq_lens_cpu = torch.tensor(all_seq_lens, dtype=torch.int64)
    batch.prefix_lens = all_prefix_lens
    batch.extend_lens = all_extend_lens

    # ============ KV cache 分配 ============
    # Evict if needed - 检查物理空闲空间
    if self.token_allocator.available_size() < total_new_tokens and self.prefix_cache is not None:
      needed_to_evict = total_new_tokens - self.token_allocator.available_size()
      needed_to_evict = (
        (needed_to_evict + self.page_size - 1) // self.page_size * self.page_size
      )  # 向上对齐到 page_size
      logger.debug(
        "prepare_for_mixed: need to evict for extend batch, "
        f"extend_num_tokens={total_new_tokens}, available_size={self.token_allocator.available_size()}, needed_to_evict={needed_to_evict}"
      )
      self.prefix_cache.evict(needed_to_evict)

    # --- Extend 请求: 分配新的 req_pool slot + KV cache ---
    extend_bs = len(extend_reqs)
    extend_req_pool_indices = []
    if extend_bs > 0:
      pool_indices = self.request_pool.alloc(extend_bs)
      if pool_indices is None:
        raise RuntimeError(
          f"Failed to allocate request pool slots for extend requests: "
          f"requested {extend_bs} slots, but request pool is full. "
          f"This usually indicates a scheduling bug where batch size exceeds max_num_seqs."
        )
      for req, pool_idx in zip(extend_reqs, pool_indices):
        req.req_pool_idx = int(pool_idx)
      extend_req_pool_indices = list(pool_indices)

      last_loc = [
        (t[-1:] if len(t) > 0 else torch.tensor([-1], device=self.device))
        for t in extend_prefix_tensors
      ]

      extend_out_cache_loc = self.token_allocator.alloc_pages_extend(
        prefix_lens=torch.tensor(extend_prefix_lens, dtype=torch.int64).to(self.device),
        prefix_lens_cpu=torch.tensor(extend_prefix_lens, dtype=torch.int64),
        seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int64).to(self.device),
        seq_lens_cpu=torch.tensor(extend_seq_lens, dtype=torch.int64),
        last_loc=torch.cat(last_loc),
        extend_num_tokens=extend_num_tokens,
      )
      if extend_out_cache_loc is None:
        physical_available = self.token_allocator.available_size()
        logical_available = self.available_tokens()
        raise RuntimeError(
          f"Failed to allocate KV cache for mixed batch extend part: "
          f"need {extend_num_tokens} tokens, "
          f"physical_available {physical_available} tokens (free pages), "
          f"logical_available {logical_available} tokens (includes evictable radix)"
        )

      # Write prefix + extend indices to request pool
      extend_req_pool_indices_tensor = torch.tensor(extend_req_pool_indices, dtype=torch.int64).to(
        self.device
      )
      prefix_lens_device = torch.tensor(extend_prefix_lens, dtype=torch.int64).to(self.device)
      extend_lens_device = torch.tensor(extend_extend_lens, dtype=torch.int64).to(self.device)
      extend_seq_lens_tensor = torch.tensor(extend_seq_lens, dtype=torch.int64).to(self.device)

      write_cache_indices(
        extend_out_cache_loc,
        extend_req_pool_indices_tensor,
        prefix_lens_device,
        extend_seq_lens_tensor,
        extend_lens_device,
        extend_prefix_tensors,
        self.request_pool,
      )
    else:
      extend_out_cache_loc = torch.tensor([], dtype=torch.int64, device=self.device)

    # --- Decode 请求: 已有 req_pool slot，只分配 1 个新 token 的 KV 槽位 ---
    decode_req_pool_indices = []
    if decode_reqs:
      decode_req_pool_indices = [r.req_pool_idx for r in decode_reqs]
      decode_rpi = torch.tensor(decode_req_pool_indices, dtype=torch.int64).to(self.device)
      # seq_lens for decode reqs before +1 (current seq len)
      decode_seq_lens_tensor = torch.tensor([s - 1 for s in decode_seq_lens], dtype=torch.int64).to(
        self.device
      )
      decode_seq_lens_cpu = torch.tensor([s - 1 for s in decode_seq_lens], dtype=torch.int64)

      last_loc = self.request_pool.req_to_token_pool()[decode_rpi, decode_seq_lens_tensor - 1]
      decode_out_cache_loc = self.token_allocator.alloc_pages_decode(
        seq_lens=decode_seq_lens_tensor,
        seq_lens_cpu=decode_seq_lens_cpu,
        last_loc=last_loc,
      )

      # Update seq_lens and request pool mapping for decode tokens
      new_pos = decode_seq_lens_tensor  # position = old_seq_len (0-indexed)
      self.request_pool.req_to_token_pool()[decode_rpi, new_pos] = decode_out_cache_loc.to(
        torch.int32
      )
    else:
      decode_out_cache_loc = torch.tensor([], dtype=torch.int64, device=self.device)

    # ============ 合并所有 out_cache_loc 和 req_pool_indices ============
    all_req_pool_indices = extend_req_pool_indices + decode_req_pool_indices
    batch.req_pool_indices = torch.tensor(all_req_pool_indices, dtype=torch.int64).to(self.device)
    batch.out_cache_loc = torch.cat([extend_out_cache_loc, decode_out_cache_loc])

  def prepare_for_decode(self, batch: "ScheduledBatch"):
    batch.forward_mode = ForwardMode.DECODE
    # Decode 阶段的 input_ids 是每个请求最后生成的 token
    # 从每个 req.output_ids[-1] 获取
    bs = len(batch.reqs)
    if bs == 0:
      return
    last_tokens = [req.output_ids[-1] for req in batch.reqs]
    batch.input_ids = torch.tensor(last_tokens, dtype=torch.int64).to(self.device)
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
    new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)

    # 注意：这里不释放内存！
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
