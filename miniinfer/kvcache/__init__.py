"""
KV Cache 模块

包含:
- KVCacheManager: KV Cache 管理器
- MemoryBudgetManager: 显存预算管理器
- PrefillAdder: Prefill 预算控制器（支持 chunked prefill）
- TokenBudgetAdder: Token 预算加法器（旧版兼容）
- MHAKVCacheStorage, PagedTokenAllocator, RequestPool: 底层存储组件
"""

from .kv_cache_manager import KVCacheManager
from .memory_budget import (
    MemoryBudgetManager,
    PrefillAdder,
    TokenBudgetAdder,
    get_gpu_memory_stats,
    estimate_kv_cache_memory_per_token,
    MemoryStats,
)
from .memory_pool import (
    MHAKVCacheStorage,
    PagedTokenAllocator,
    TokenAllocator,
    RequestPool,
)
from .interface import (
    IKVCacheStorage,
    ITokenAllocator,
    IRequestPool,
    IPrefixCache,
)

__all__ = [
    # Manager
    "KVCacheManager",
    # Memory Budget
    "MemoryBudgetManager",
    "PrefillAdder",
    "TokenBudgetAdder",
    "get_gpu_memory_stats",
    "estimate_kv_cache_memory_per_token",
    "MemoryStats",
    # Storage
    "MHAKVCacheStorage",
    "PagedTokenAllocator",
    "TokenAllocator",
    "RequestPool",
    # Interfaces
    "IKVCacheStorage",
    "ITokenAllocator",
    "IRequestPool",
    "IPrefixCache",
]
