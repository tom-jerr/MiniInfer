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
        size: int,  # 现在可以为 None，自动计算
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

        logger.info(
            f"KVCacheManager: size={self.size} tokens, "
            f"num_pages={self.memory_budget.num_pages}, "
            f"memory={self.memory_budget.kv_cache_memory_bytes / (1024**3):.2f}GB"
        )

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
            req.prefix_indices = torch.empty(
                (0,), dtype=torch.int64, device=self.device
            )
            req.cache_protected_len = 0
            req.last_node = None
        req.extend_input_len = len(req.fill_ids) - req.cache_protected_len
        print(
            f"prefix_for_waiting_req: req_id={req.req_id}, prefix_len={req.cache_protected_len}, extend_input_len={req.extend_input_len}"
        )

    def prepare_for_extend(
        self,
        batch: "ScheduledBatch",
    ):
        """batch is mutable"""

        # Init batch metadata
        batch.forward_mode = ForwardMode.EXTEND
        extend_ids = [r.fill_ids[len(r.prefix_indices):] for r in batch.reqs]
        extend_num_tokens = sum(len(ids) for ids in extend_ids)
        seq_lens = [len(r.fill_ids) for r in batch.reqs]
        prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
        extend_lens = [r.extend_input_len for r in batch.reqs]

        extend_ids_tensor = torch.tensor(
            [token_id for ids in extend_ids for token_id in ids], dtype=torch.int64
        ).to(self.device)
        seq_lens_tensor = torch.tensor(
            seq_lens, dtype=torch.int64).to(self.device)
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
        # Persist request-pool indices on the Req objects for correct release/update flows.
        if req_pool_indices is not None:
            for req, pool_idx in zip(batch.reqs, req_pool_indices):
                req.req_pool_idx = int(pool_idx)
        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            self.device
        )

        # Allocate KV cache slots
        if (
            self.available_tokens() < extend_num_tokens
            and self.prefix_cache is not None
        ):
            self.prefix_cache.evict(extend_num_tokens)

        last_loc = [
            (t[-1:] if len(t) > 0 else torch.tensor([-1], device=self.device))
            for t in prefix_tensors
        ]
        # torch.cat produces shape (bs,) which kernel expects; torch.stack would give (bs, 1)
        out_cache_loc = self.token_allocator.alloc_pages_extend(
            prefix_lens=torch.tensor(batch.prefix_lens, dtype=torch.int64).to(
                self.device
            ),
            prefix_lens_cpu=torch.tensor(batch.prefix_lens, dtype=torch.int64),
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            last_loc=torch.cat(last_loc),
            extend_num_tokens=extend_num_tokens,
        )

        # 检查分配是否成功
        if out_cache_loc is None:
            raise RuntimeError(
                f"Failed to allocate KV cache for extend batch: "
                f"need {extend_num_tokens} tokens ({bs} requests), "
                f"available {self.available_tokens()} tokens"
            )

        # Write prefix cache and new extend cache to request pool
        prefix_lens_device = torch.tensor(prefix_lens, dtype=torch.int64).to(
            self.device
        )
        extend_lens_device = torch.tensor(extend_lens, dtype=torch.int64).to(
            self.device
        )
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

    def prepare_for_decode(self, batch: "ScheduledBatch"):
        batch.forward_mode = ForwardMode.DECODE
        # Decode 阶段的 input_ids 是每个请求最后生成的 token
        # 从每个 req.output_ids[-1] 获取
        bs = len(batch.reqs)
        if bs == 0:
            return
        last_tokens = [req.output_ids[-1] for req in batch.reqs]
        batch.input_ids = torch.tensor(
            last_tokens, dtype=torch.int64).to(self.device)
        batch.output_ids = None
        token_per_req = 1  # decode

        # Allocate KV cache slots
        if (
            self.available_tokens() < bs * token_per_req
            and self.prefix_cache is not None
        ):
            self.prefix_cache.evict(bs * token_per_req)

        last_loc = self.request_pool.req_to_token_pool()[
            batch.req_pool_indices, batch.seq_lens - 1
        ]
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
        self.request_pool.req_to_token_pool()[batch.req_pool_indices, new_pos] = (
            out_cache_loc.to(torch.int32)
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
        req_pool_idx = (
            int(req.req_pool_idx)
            if getattr(req, "req_pool_idx", -1) >= 0
            else int(req.req_id)
        )
        token_ids = req.origin_input_ids + req.output_ids
        num_tokens = len(token_ids)
        kv_indices = self.request_pool.read(req_pool_idx, slice(0, num_tokens))
        if is_insert and self.prefix_cache is not None:
            # 插入到前缀缓存
            new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)
            self.token_allocator.free(
                kv_indices[req.cache_protected_len: new_prefix_len]
            )
        else:
            self.token_allocator.free(kv_indices[req.cache_protected_len:])

        self.request_pool.free([req_pool_idx])  # param is list
        if self.prefix_cache is not None:
            self.prefix_cache.dec_lock_ref(req.last_node)

    def update_finished_req_radix_cache(self, req: Req, is_insert: bool = True):
        """更新前缀缓存"""
        if is_insert and self.prefix_cache is not None:
            token_ids = req.origin_input_ids + req.output_ids
            num_tokens = len(token_ids)
            kv_indices = self.request_pool.read(
                req.req_pool_idx, slice(0, num_tokens))
            new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)
            self.token_allocator.free(
                kv_indices[req.cache_protected_len: new_prefix_len]
            )
        else:
            self.token_allocator.free(kv_indices[req.cache_protected_len:])

        self.request_pool.free([req.req_pool_idx])  # param is list
        if self.prefix_cache is not None:
            self.prefix_cache.dec_lock_ref(req.last_node)

    def update_unfinished_req_radix_cache(self, req: Req):
        """更新前缀缓存"""
        if self.prefix_cache is None:
            return
        token_ids = req.fill_ids
        kv_indices = self.request_pool.read(
            req.req_pool_idx, slice(0, len(token_ids)))
        new_prefix_len = self.prefix_cache.insert(token_ids, kv_indices)
        self.token_allocator.free(
            kv_indices[req.cache_protected_len: new_prefix_len])
        # update req metadata
        new_indices, new_last_node = self.prefix_cache.match_prefix(token_ids)
        self.request_pool.write(
            req.req_pool_idx,
            slice(req.cache_protected_len, len(new_indices)),
            new_indices[req.cache_protected_len:],
        )

        self.prefix_cache.dec_lock_ref(req.last_node)
        self.prefix_cache.inc_lock_ref(new_last_node)

        req.cache_protected_len = len(new_indices)
        req.prefix_indices = new_indices
        req.last_node = new_last_node

    # =============== Attention forward calling ================
    def get_page_table(
        self, forward_batch: ForwardBatch, seq_len: torch.Tensor
    ) -> torch.Tensor:
        """获取指定 batch 的 page table"""
        return self.request_pool.req_to_token_pool()[
            forward_batch.req_pool_indices, :seq_len
        ]

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
        return self.token_allocator.available_size() + self.prefix_cache.evictable_size() if self.prefix_cache is not None else self.token_allocator.available_size()

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
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + offset
            + pre_len,
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
