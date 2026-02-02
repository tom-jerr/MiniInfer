"""
Memory Budget Manager - 管理 GPU 显存预算和 token 分配

职责:
1. 计算 GPU 可用显存
2. 根据配置的利用率百分比确定可用的 token 预算
3. 为调度器提供 token 预算限制
4. 支持 chunked prefill 的预算控制
"""

from __future__ import annotations
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from miniinfer.engine.scheduler_batch import Req
    from miniinfer.kvcache.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


@dataclass
class MemoryStats:
    """GPU 显存统计信息"""

    total_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    @property
    def allocated_gb(self) -> float:
        return self.allocated_bytes / GB

    @property
    def free_gb(self) -> float:
        return self.free_bytes / GB

    def __repr__(self) -> str:
        return (
            f"MemoryStats(total={self.total_gb:.2f}GB, "
            f"allocated={self.allocated_gb:.2f}GB, "
            f"free={self.free_gb:.2f}GB)"
        )


def get_gpu_memory_stats(device: str = "cuda") -> MemoryStats:
    """获取 GPU 显存统计信息"""
    if not torch.cuda.is_available():
        return MemoryStats(0, 0, 0, 0)

    device_idx = (
        torch.cuda.current_device() if device == "cuda" else int(device.split(":")[1])
    )
    total = torch.cuda.get_device_properties(device_idx).total_memory
    allocated = torch.cuda.memory_allocated(device_idx)
    reserved = torch.cuda.memory_reserved(device_idx)
    # 实际可用 = 总量 - 已分配
    # 注：reserved 包含了 PyTorch 的缓存，可能比 allocated 大
    free = total - reserved

    return MemoryStats(
        total_bytes=total,
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        free_bytes=free,
    )


def estimate_kv_cache_memory_per_token(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    估算每个 token 的 KV cache 显存占用（字节）

    每层每个 token 需要存储:
    - K: [num_kv_heads, head_dim]
    - V: [num_kv_heads, head_dim]
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    # K + V per layer
    bytes_per_layer = 2 * num_kv_heads * head_dim * element_size
    return bytes_per_layer * num_layers


class MemoryBudgetManager:
    """
    显存预算管理器

    根据 GPU 可用显存和配置的利用率，计算可用的 token 预算。
    用于限制调度器的 token 分配，防止 OOM。
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        gpu_memory_utilization: float = 0.9,
        max_total_tokens: int = 20480,
        max_num_batched_tokens: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        初始化显存预算管理器

        Args:
            num_layers: 模型层数
            num_kv_heads: KV head 数量
            head_dim: head 维度
            gpu_memory_utilization: GPU 显存利用率 (0.0-1.0)
            max_total_tokens: KV cache 的最大 token 数（来自 EngineConfig）
            max_num_batched_tokens: 每个 batch 的最大 token 数（用于 chunked prefill）
            dtype: 数据类型
            device: 设备
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_total_tokens = max_total_tokens
        self.dtype = dtype
        self.device = device

        # 计算每个 token 的 KV cache 占用
        self.bytes_per_token = estimate_kv_cache_memory_per_token(
            num_layers, num_kv_heads, head_dim, dtype
        )

        # 设置每个 batch 的最大 token 数
        # 如果没有指定，使用 max_total_tokens 作为默认值
        self.max_num_batched_tokens = max_num_batched_tokens or max_total_tokens

        # 初始化时获取显存状态
        self._update_memory_stats()

        logger.info(
            f"MemoryBudgetManager initialized: "
            f"bytes_per_token={self.bytes_per_token}, "
            f"max_num_batched_tokens={self.max_num_batched_tokens}, "
            f"gpu_utilization={gpu_memory_utilization}"
        )

    def _update_memory_stats(self):
        """更新显存统计信息"""
        self.memory_stats = get_gpu_memory_stats(self.device)

    def get_available_memory_bytes(self) -> int:
        """
        获取可用于 KV cache 的显存字节数

        考虑 gpu_memory_utilization 配置
        """
        self._update_memory_stats()
        # 基于总显存和利用率计算可用显存
        usable_memory = int(self.memory_stats.total_bytes * self.gpu_memory_utilization)
        # 减去当前已分配的显存
        available = usable_memory - self.memory_stats.allocated_bytes
        return max(0, available)

    def estimate_tokens_from_memory(self, memory_bytes: int) -> int:
        """根据显存字节数估算可容纳的 token 数量"""
        if self.bytes_per_token == 0:
            return 0
        return memory_bytes // self.bytes_per_token

    def estimate_memory_from_tokens(self, num_tokens: int) -> int:
        """根据 token 数量估算需要的显存字节数"""
        return num_tokens * self.bytes_per_token

    def get_max_new_tokens_budget(self, currently_used_tokens: int = 0) -> int:
        """
        获取当前可以新分配的最大 token 数

        Args:
            currently_used_tokens: 当前已使用的 token 数（运行中的请求）

        Returns:
            可以新分配的最大 token 数
        """
        # 方法1：基于 max_total_tokens 配置
        budget_from_config = self.max_total_tokens - currently_used_tokens

        # 方法2：基于实际可用显存
        available_memory = self.get_available_memory_bytes()
        budget_from_memory = self.estimate_tokens_from_memory(available_memory)

        # 取两者的最小值
        budget = min(budget_from_config, budget_from_memory)

        return max(0, budget)

    def get_batch_token_budget(self) -> int:
        """
        获取单个 batch 的 token 预算

        这是 chunked prefill 的核心：限制每个 batch 处理的 token 总数
        """
        return self.max_num_batched_tokens

    def can_allocate(self, num_tokens: int, current_usage: int = 0) -> bool:
        """
        检查是否可以分配指定数量的 tokens

        Args:
            num_tokens: 需要分配的 token 数
            current_usage: 当前已使用的 token 数

        Returns:
            是否可以分配
        """
        budget = self.get_max_new_tokens_budget(current_usage)
        return num_tokens <= budget

    def get_stats(self) -> dict:
        """获取统计信息"""
        self._update_memory_stats()
        return {
            "memory_stats": self.memory_stats,
            "bytes_per_token": self.bytes_per_token,
            "max_total_tokens": self.max_total_tokens,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "estimated_max_tokens_from_memory": self.estimate_tokens_from_memory(
                self.get_available_memory_bytes()
            ),
        }


@dataclass
class PrefillAdder:
    """
    Prefill 预算控制器 - 支持 Chunked Prefill

    核心策略:
    1. 优先保证 decode 请求：reserved_size = running_bs（正在运行的 decode 请求数）
    2. prefill_budget = max_extend_tokens - reserved_size
    3. 如果请求太大无法完整 prefill，则分块处理
    4. 分块的请求不会进入 decode 阶段，继续留在 waiting queue

    使用方式:
        adder = PrefillAdder(
            prefill_budget=8192,
            reserved_size=running_bs,
            kv_cache_mgr=kv_cache_mgr,
        )
        # 先处理 decode
        for req in running_reqs:
            adder.add_decode(req)
        # 再处理 prefill
        for req in waiting_queue:
            result = adder.try_add_prefill(req)
            if result is None:
                break
    """

    prefill_budget: int  # 可用于 prefill 的 token 预算
    reserved_size: int  # 为 decode 预留的 token 数（= running_bs）
    kv_cache_mgr: "KVCacheManager"  # KV cache 管理器
    max_batch_size: int = 512  # 最大 batch size

    # 运行时状态（使用 field(default_factory=...) 初始化可变对象）
    current_prefill_tokens: int = field(default=0, init=False)
    current_decode_tokens: int = field(default=0, init=False)
    current_prefill_pages: int = field(default=0, init=False)  # 追踪累计的 prefill 页数
    prefill_reqs: List = field(default_factory=list, init=False)
    decode_reqs: List = field(default_factory=list, init=False)
    chunked_reqs: List = field(default_factory=list, init=False)  # 被分块的请求

    def __post_init__(self):
        """初始化后重置状态"""
        self.reset()

    def reset(self):
        """重置状态，开始新的 batch 调度"""
        self.current_prefill_tokens = 0
        self.current_decode_tokens = 0
        self.current_prefill_pages = 0
        self.prefill_reqs = []
        self.decode_reqs = []
        self.chunked_reqs = []

    @property
    def total_tokens(self) -> int:
        """当前总 token 数"""
        return self.current_prefill_tokens + self.current_decode_tokens

    @property
    def total_reqs(self) -> int:
        """当前总请求数"""
        return len(self.prefill_reqs) + len(self.decode_reqs)

    @property
    def available_prefill_budget(self) -> int:
        """剩余可用于 prefill 的预算"""
        return self.prefill_budget - self.current_prefill_tokens

    def _get_page_size(self) -> int:
        """获取 page size，用于分页对齐计算"""
        page_size = getattr(self.kv_cache_mgr, "page_size", None)
        if page_size is None or not isinstance(page_size, int):
            return 1
        return page_size

    def _estimate_pages_needed(self, num_tokens: int, prefix_len: int = 0) -> int:
        """估算需要的新页数

        对于 paged KV cache，需要计算请求需要多少新页，而不是多少 tokens。

        Args:
            num_tokens: 需要分配的 token 数（extend_len）
            prefix_len: 已有的前缀长度

        Returns:
            需要的新页数
        """
        page_size = self._get_page_size()
        if page_size <= 1:
            return num_tokens

        # 计算扩展后的总长度需要的页数
        total_len = prefix_len + num_tokens
        pages_after = (total_len + page_size - 1) // page_size

        # 计算之前已有的页数
        pages_before = (
            (prefix_len + page_size - 1) // page_size if prefix_len > 0 else 0
        )

        # 需要的新页数
        return pages_after - pages_before

    def _can_allocate_kv_cache(self, num_tokens: int, prefix_len: int = 0) -> bool:
        """检查 KV cache 是否有足够的空间

        需要考虑：
        1. 当前批次中已经预分配的 prefill 页数（current_prefill_pages）
        2. 当前批次中 decode 需要的新页数（通过 _decode_pages_needed 计算）
        3. 分页对齐：实际分配按页进行
        """
        page_size = self._get_page_size()

        # 对于非分页分配器，直接按 token 计算
        if page_size <= 1:
            effective_available = (
                self.kv_cache_mgr.available_tokens()
                - self.current_prefill_tokens
                - self.current_decode_tokens
            )
            return effective_available >= num_tokens

        # 对于分页分配器，需要按页计算
        # available_tokens() 返回的是 free_pages * page_size
        available_pages = self.kv_cache_mgr.available_tokens() // page_size

        # 计算 decode 需要的新页数
        decode_pages = self._decode_pages_needed()

        # 使用精确的页数追踪（包括 prefill 和 decode 需要的页）
        used_pages = self.current_prefill_pages + decode_pages

        # 计算这个请求需要的新页数
        pages_needed = self._estimate_pages_needed(num_tokens, prefix_len)

        # 可用页数
        remaining_pages = available_pages - used_pages

        return remaining_pages >= pages_needed

    def _decode_pages_needed(self) -> int:
        """计算当前 decode 请求需要的新页数

        这需要在所有 decode 请求添加后调用，用于估算 prefill 的可用空间。
        """
        page_size = self._get_page_size()
        if page_size <= 1:
            return 0

        count = 0
        for req in self.decode_reqs:
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            # decode 后 seq_len 变为 seq_len + 1
            # 如果 (seq_len + 1) % page_size == 1，说明新 token 是新页的第一个 token
            if (seq_len + 1) % page_size == 1:
                count += 1
        return count

    def add_decode(self, req: "Req") -> bool:
        """
        添加一个 decode 请求

        Decode 请求优先级最高，每个请求消耗 1 token。
        注意：Decode 请求总是应该被接受，因为它们的 KV cache 已经被分配。
        拒绝 decode 会导致死锁（KV cache 无法释放）。

        Args:
            req: 请求对象

        Returns:
            是否成功添加
        """
        if self.total_reqs >= self.max_batch_size:
            return False

        self.current_decode_tokens += 1
        self.decode_reqs.append(req)
        return True

    def try_add_prefill(
        self,
        req: "Req",
        chunk_size: Optional[int] = None,
    ) -> Optional[Tuple["Req", int, bool]]:
        """
        尝试添加一个 prefill 请求

        策略:
        1. 如果请求可以完整 prefill，直接添加
        2. 如果请求太大但启用了 chunked prefill，分块处理
        3. 分块的请求返回 (req, actual_chunk_size, is_chunked=True)

        Args:
            req: 请求对象（已经调用过 prefix_for_waiting_req）
            chunk_size: 可选的 chunk 大小限制

        Returns:
            None: 无法添加（预算不足或 KV cache 满）
            (req, extend_len, is_chunked): 成功添加
        """
        if self.total_reqs >= self.max_batch_size:
            return None

        extend_len = req.extend_input_len
        # 获取前缀长度，优先使用 cache_protected_len
        prefix_len = getattr(req, "cache_protected_len", 0) or 0
        if prefix_len == 0:
            prefix_indices = getattr(req, "prefix_indices", None)
            if prefix_indices is not None and hasattr(prefix_indices, "__len__"):
                prefix_len = len(prefix_indices)

        # 检查是否可以完整 prefill
        if extend_len <= self.available_prefill_budget:
            # 检查 KV cache 容量（考虑分页开销）
            if not self._can_allocate_kv_cache(extend_len, prefix_len):
                page_size = self._get_page_size()
                available_pages = (
                    self.kv_cache_mgr.available_tokens() // page_size
                    if page_size > 0
                    else 0
                )
                logger.warning(
                    f"KV cache insufficient for req {req.req_id}: "
                    f"need {extend_len} tokens ({self._estimate_pages_needed(extend_len, prefix_len)} pages), "
                    f"current_prefill_pages={self.current_prefill_pages}, "
                    f"available_pages={available_pages}"
                )
                return None

            # 更新 token 和页数计数
            pages_needed = self._estimate_pages_needed(extend_len, prefix_len)
            self.current_prefill_tokens += extend_len
            self.current_prefill_pages += pages_needed
            self.prefill_reqs.append(req)
            return (req, extend_len, False)

        # 尝试 chunked prefill
        if chunk_size is None:
            chunk_size = self.available_prefill_budget

        if chunk_size <= 0:
            return None

        # 实际可以处理的 chunk 大小
        actual_chunk_size = min(chunk_size, extend_len, self.available_prefill_budget)

        if actual_chunk_size <= 0:
            return None

        # 检查 KV cache 容量（chunked prefill 也需要考虑 prefix_len）
        if not self._can_allocate_kv_cache(actual_chunk_size, prefix_len):
            return None

        # 标记为 chunked 请求，更新 token 和页数计数
        pages_needed = self._estimate_pages_needed(actual_chunk_size, prefix_len)
        self.current_prefill_tokens += actual_chunk_size
        self.current_prefill_pages += pages_needed
        self.chunked_reqs.append((req, actual_chunk_size))
        return (req, actual_chunk_size, True)

    def get_batch_summary(self) -> dict:
        """获取当前 batch 的摘要"""
        return {
            "prefill_reqs": len(self.prefill_reqs),
            "decode_reqs": len(self.decode_reqs),
            "chunked_reqs": len(self.chunked_reqs),
            "prefill_tokens": self.current_prefill_tokens,
            "prefill_pages": self.current_prefill_pages,
            "decode_tokens": self.current_decode_tokens,
            "total_tokens": self.total_tokens,
            "prefill_budget_remaining": self.available_prefill_budget,
        }


# 保留旧的 TokenBudgetAdder 作为别名以保持向后兼容
class TokenBudgetAdder:
    """
    Token 预算加法器 - 用于 Chunked Prefill

    注意: 这是旧版实现，建议使用 PrefillAdder
    """

    def __init__(
        self,
        token_budget: int,
        max_batch_size: int,
        reserved_decode_tokens: int = 0,
    ):
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.reserved_decode_tokens = reserved_decode_tokens
        self.reset()

    def reset(self):
        self.current_extend_tokens = 0
        self.current_decode_tokens = 0
        self.extend_reqs = []
        self.decode_reqs = []

    @property
    def total_tokens(self) -> int:
        return self.current_extend_tokens + self.current_decode_tokens

    @property
    def total_reqs(self) -> int:
        return len(self.extend_reqs) + len(self.decode_reqs)

    @property
    def available_budget(self) -> int:
        return self.token_budget - self.total_tokens

    def can_add_extend(self, extend_len: int, prefix_len: int = 0) -> bool:
        if self.total_reqs >= self.max_batch_size:
            return False
        if extend_len > self.available_budget - self.reserved_decode_tokens:
            return False
        return True

    def add_extend(self, req, extend_len: int) -> bool:
        if not self.can_add_extend(extend_len):
            return False
        self.current_extend_tokens += extend_len
        self.extend_reqs.append(req)
        return True

    def can_add_decode(self) -> bool:
        if self.total_reqs >= self.max_batch_size:
            return False
        if self.total_tokens + 1 > self.token_budget:
            return False
        return True

    def add_decode(self, req) -> bool:
        if not self.can_add_decode():
            return False
        self.current_decode_tokens += 1
        self.decode_reqs.append(req)
        return True

    def get_batch_summary(self) -> dict:
        return {
            "extend_reqs": len(self.extend_reqs),
            "decode_reqs": len(self.decode_reqs),
            "extend_tokens": self.current_extend_tokens,
            "decode_tokens": self.current_decode_tokens,
            "total_tokens": self.total_tokens,
            "budget_remaining": self.available_budget,
        }
