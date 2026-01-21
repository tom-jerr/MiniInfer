"""KV Cache Manager - 由 Scheduler 持有"""

from typing import Optional, List, Tuple, Any
import torch
import logging

from kvcache.interface import (
    IKVCacheStorage,
    ITokenAllocator,
    IRequestPool,
    IPrefixCache,
)
from kvcache.memory_pool import MHAKVCacheStorage, TokenAllocator, RequestPool
from kvcache.radix_cache import RadixCache
from engine.scheduler_batch import ScheduledBatch, Req, ForwardMode, ForwardBatch

logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    KV Cache 管理器

    由多个组件构成:
    - CacheStorage: 物理存储
    - TokenAllocator: 分配 KV indices
    - RequestPool: 管理不同请求分配
    - PrefixCache: 前缀缓存 (Radix Tree)
    """

    def __init__(
        self,
        size: int,
        max_requests: int,
        max_context_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        enable_prefix_cache: bool = True,
        page_size: int = 1,
    ):
        self.size = size
        self.max_requests = max_requests
        self.max_context_len = max_context_len
        self.device = device
        self.enable_prefix_cache = enable_prefix_cache
        self.page_size = page_size

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
        self.token_allocator: ITokenAllocator = TokenAllocator(
            size=size,
            device=device,
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

        logger.info(
            f"KVCacheManager initialized: size={size}, "
            f"max_requests={max_requests}, prefix_cache={enable_prefix_cache}"
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
        req.extend_input_len = len(req.fill_ids) - req.cache_protected_len

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
        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            self.device
        )

        # Allocate KV cache slots
        if self.available_tokens() < extend_num_tokens:
            self.prefix_cache.evict(extend_num_tokens)
        out_cache_loc = self.token_allocator.alloc(extend_num_tokens)

        # Write prefix cache and new extend cache to request pool
        for i in range(bs):
            req = batch.reqs[i]
            req_idx = req_pool_indices[i]
            req.req_pool_idx = req_idx
            self.request_pool.write(
                req_idx=req_idx,
                token_range=slice(0, batch.prefix_lens[i]),
                kv_indices=prefix_tensors[i],
            )
            self.request_pool.write(
                req_idx=req_idx,
                token_range=slice(
                    batch.prefix_lens[i], batch.prefix_lens[i] + batch.extend_lens[i]
                ),
                kv_indices=out_cache_loc[
                    sum(batch.extend_lens[:i]) : sum(batch.extend_lens[: i + 1])
                ],
            )

        batch.req_pool_indices = req_pool_indices_tensor
        batch.out_cache_loc = out_cache_loc

    def prepare_for_decode(
        self, batch: "ScheduledBatch"
    ):
        batch.forward_mode = ForwardMode.DECODE
        batch.input_ids = batch.output_ids
        batch.output_ids = None
        bs = len(batch.reqs)
        if bs == 0:
            return
        token_per_req = 1 # decode

        # Allocate KV cache slots
        if self.available_tokens() < bs * token_per_req:
            self.prefix_cache.evict(bs * token_per_req)
        out_cache_loc = self.token_allocator.alloc(bs * token_per_req)
        for i in range(bs):
            req = batch.reqs[i]
            req_idx = req.req_idx
            self.request_pool.write(
                req_idx=req_idx,
                token_range=slice(req.fill_ids, req.fill_ids + token_per_req),
                kv_indices=out_cache_loc[i * token_per_req : (i + 1) * token_per_req],
            )
        batch.out_cache_loc = out_cache_loc
        batch.seq_lens.add_(1)
        batch.seq_lens_cpu.add_(1)
        

    def release_request(
        self,
        req:Req,
        is_insert: bool=True,
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
        req_idx = req.req_id
        token_ids = req.origin_input_ids + req.output_ids
        num_tokens = len(token_ids)
        kv_indices = self.request_pool.read(req_idx, slice(0, num_tokens))
        if is_insert and self.prefix_cache is not None:
            # 插入到前缀缓存
            new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)
            self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])
        else:
            self.token_allocator.free(kv_indices[req.cache_protected_len: ])

        self.request_pool.free(req_idx)
        self.prefix_cache.dec_lock_ref(req.last_node)

    def update_finished_req_radix_cache(self, req: Req, is_insert: bool=True):
        """更新前缀缓存"""
        if is_insert and self.prefix_cache is not None:
            token_ids = req.origin_input_ids + req.output_ids
            num_tokens = len(token_ids)
            kv_indices = self.request_pool.read(req.req_pool_idx, slice(0, num_tokens))
            new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)
            self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])
        else:
            self.token_allocator.free(kv_indices[req.cache_protected_len: ])
        
        self.request_pool.free(req.req_pool_idx)
        self.prefix_cache.dec_lock_ref(req.last_node)

    def update_unfinished_req_radix_cache(self, req: Req):
        """更新前缀缓存"""
        if self.prefix_cache is None:
            return
        token_ids = req.fill_ids
        kv_indices = self.request_pool.read(req.req_pool_idx, slice(0, len(token_ids)))
        new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)
        self.token_allocator.free(kv_indices[req.cache_protected_len : new_prefix_len])
        # update req metadata
        new_indices, new_last_node = self.prefix_cache.match_prefix(token_ids)
        self.request_pool.write(
            req.req_pool_idx, slice(req.cache_protected_len, len(new_indices)), new_indices[req.cache_protected_len :]) 
        
        self.prefix_cache.dec_lock_ref(req.last_node)
        self.prefix_cache.inc_lock_ref(new_last_node)

        req.cache_protected_len = len(new_indices)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    # =============== Attention forward calling ================
    def get_page_table(self, forward_batch: ForwardBatch, seq_len: torch.Tensor) -> torch.Tensor:
        """获取指定 batch 的 page table"""
        return self.request_pool.req_to_token_pool()[forward_batch.req_pool_indices,: seq_len]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 KV buffer"""
        return self.storage.get_kv_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """写入 KV cache"""
        self.storage.set_kv_buffer(layer_id, loc, cache_k, cache_v)

    def available_tokens(self) -> int:
        """返回可用的 token 槽位数"""
        return self.token_allocator.available_size()

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
