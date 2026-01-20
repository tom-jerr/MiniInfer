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

    def alloc_for_request(
        self,
        req_idx: int,
        token_ids: List[int],
        num_new_tokens: int,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        为请求分配 KV cache

        尝试匹配前缀缓存，然后分配新的 tokens

        Args:
            req_idx: 请求索引
            token_ids: 完整的 token ids
            num_new_tokens: 需要新分配的 token 数量

        Returns:
            (new_kv_indices, num_cached_tokens)
        """
        cached_indices = None
        num_cached = 0
        last_node = None

        # 尝试匹配前缀缓存
        if self.prefix_cache is not None:
            cached_indices, last_node = self.prefix_cache.match_prefix(token_ids)
            num_cached = len(cached_indices)

            if num_cached > 0:
                # 锁定缓存节点，防止被驱逐
                self.prefix_cache.inc_lock_ref(last_node)
                # 写入缓存的映射
                self.request_pool.write(req_idx, slice(0, num_cached), cached_indices)

        # 计算实际需要分配的 token 数量
        actual_new_tokens = num_new_tokens - num_cached
        if actual_new_tokens <= 0:
            return cached_indices, num_cached

        # 分配新的 KV cache 空间
        new_indices = self.alloc_tokens(actual_new_tokens)
        if new_indices is None:
            # 空间不足，尝试驱逐缓存
            if self.prefix_cache is not None:
                self.prefix_cache.evict(actual_new_tokens)
                new_indices = self.alloc_tokens(actual_new_tokens)

        if new_indices is not None:
            # 写入新分配的映射
            start_pos = num_cached
            end_pos = num_cached + len(new_indices)
            self.request_pool.write(req_idx, slice(start_pos, end_pos), new_indices)

        return new_indices, num_cached

    def release_request(
        self,
        req_idx: int,
        token_ids: List[int],
        num_tokens: int,
        cache_to_radix: bool = True,
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
        kv_indices = self.request_pool.read(req_idx, slice(0, num_tokens))

        if cache_to_radix and self.prefix_cache is not None:
            # 插入到 radix tree
            self.prefix_cache.insert(token_ids, kv_indices)
            # 解锁节点 (insert 会创建新节点)
            _, node = self.prefix_cache.match_prefix(token_ids)
            self.prefix_cache.dec_lock_ref(node)
        else:
            # 直接释放 KV cache
            self.free_tokens(kv_indices)

        # 释放请求槽位
        self.free_request(req_idx)

    # =============== Attention forward calling ================
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

    # ============== private helper methods ==============
    def _alloc_request(self) -> Optional[int]:
        """分配一个请求槽位"""
        indices = self.request_pool.alloc(1)
        return indices[0] if indices else None

    def _free_request(self, req_idx: int):
        """释放请求槽位"""
        self.request_pool.free([req_idx])

    def _alloc_tokens(self, num_tokens: int) -> Optional[torch.Tensor]:
        """
        分配 KV cache 位置

        Args:
            num_tokens: 需要的 token 数量

        Returns:
            分配的 KV indices，如果空间不足返回 None
        """
        return self.token_allocator.alloc(num_tokens)

    def _free_tokens(self, indices: torch.Tensor):
        """释放 KV cache 位置"""
        self.token_allocator.free(indices)
