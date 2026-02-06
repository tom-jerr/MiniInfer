"""
显存预算管理模块

实现 KV Cache 显存预算计算和管理:
1. _determine_num_pages: 根据 GPU 空闲显存计算可用页面数
2. MemoryBudgetManager: 管理显存预算和 token 分配
3. PrefillAdder: Prefill 阶段的预算控制（支持 chunked prefill）
4. TokenBudgetAdder: Token 预算加法器
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any
import torch
import logging

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024
MB = 1024 * 1024


@dataclass
class MemoryStats:
    """GPU 显存统计信息"""

    total_bytes: int  # 总显存 (bytes)
    allocated_bytes: int  # 已分配显存 (bytes)
    reserved_bytes: int  # 已预留显存 (bytes)
    free_bytes: int  # 空闲显存 (bytes)

    # 兼容旧属性名
    @property
    def total(self) -> int:
        return self.total_bytes

    @property
    def allocated(self) -> int:
        return self.allocated_bytes

    @property
    def reserved(self) -> int:
        return self.reserved_bytes

    @property
    def free(self) -> int:
        return self.free_bytes

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
    """
    获取 GPU 显存统计信息

    Args:
        device: GPU 设备标识

    Returns:
        MemoryStats: 显存统计信息
    """
    if not torch.cuda.is_available():
        return MemoryStats(
            total_bytes=0, allocated_bytes=0, reserved_bytes=0, free_bytes=0
        )

    device_idx = (
        torch.cuda.current_device() if device == "cuda" else int(device.split(":")[-1])
    )

    total = torch.cuda.get_device_properties(device_idx).total_memory
    allocated = torch.cuda.memory_allocated(device_idx)
    reserved = torch.cuda.memory_reserved(device_idx)
    free = total - allocated

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
    估算每个 token 的 KV cache 内存占用

    KV cache shape per layer: [num_tokens, num_kv_heads, head_dim] * 2 (K and V)
    Total memory = 2 * num_layers * num_kv_heads * head_dim * dtype_size

    Args:
        num_layers: 模型层数
        num_kv_heads: KV head 数量 (支持 GQA/MQA)
        head_dim: head 维度
        dtype: 数据类型

    Returns:
        每个 token 的 KV cache 字节数
    """
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    # 2 for K and V
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_size
    return bytes_per_token


def estimate_kv_cache_memory_per_page(
    page_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    估算每个 page 的 KV cache 内存占用

    Args:
        page_size: 每页包含的 token 数量
        num_layers: 模型层数
        num_kv_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型

    Returns:
        每个 page 的 KV cache 字节数
    """
    bytes_per_token = estimate_kv_cache_memory_per_token(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    return bytes_per_token * page_size


def _determine_num_pages(
    page_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    memory_ratio: float = 0.9,
    num_pages_override: Optional[int] = None,
    device: str = "cuda",
) -> int:
    """
    根据 GPU 空闲显存计算可用的 KV cache 页面数量

    计算流程:
    1. 获取模型加载后的 GPU 空闲显存
    2. 计算每个 page 所需的显存: 2 * head_dim * num_kv_heads * page_size * dtype_size * num_layers
    3. 根据 memory_ratio (默认 0.9) 计算可分配给 KV cache 的显存
    4. 返回 num_pages = available_memory / memory_per_page

    Args:
        page_size: 每页的 token 数量
        num_layers: 模型层数
        num_kv_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型
        memory_ratio: GPU 显存利用率 (0.0 ~ 1.0)
        num_pages_override: 手动指定的页面数 (--num-pages)
        device: GPU 设备

    Returns:
        num_pages: 可分配的总页面数
    """
    # 如果手动指定了页面数，直接返回
    if num_pages_override is not None and num_pages_override > 0:
        logger.info(f"Using manual num_pages override: {num_pages_override}")
        return num_pages_override

    # 获取 GPU 显存统计
    memory_stats = get_gpu_memory_stats(device)
    available_memory = int(memory_stats.free_bytes * memory_ratio)

    # 计算每个 page 的显存占用
    memory_per_page = estimate_kv_cache_memory_per_page(
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # 计算页面数
    num_pages = available_memory // memory_per_page

    logger.info(
        f"KV Cache allocation: "
        f"free_memory={memory_stats.free_gb:.2f}GB, "
        f"usable={available_memory / GB:.2f}GB (ratio={memory_ratio}), "
        f"memory_per_page={memory_per_page / MB:.2f}MB, "
        f"num_pages={num_pages}"
    )

    return max(1, num_pages)  # 至少返回 1 页


def _determine_num_tokens(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    memory_ratio: float = 0.9,
    max_total_tokens_override: Optional[int] = None,
    device: str = "cuda",
) -> int:
    """
    根据 GPU 空闲显存计算可用的 KV cache token 数量

    Args:
        num_layers: 模型层数
        num_kv_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型
        memory_ratio: GPU 显存利用率 (0.0 ~ 1.0)
        max_total_tokens_override: 手动指定的最大 token 数
        device: GPU 设备

    Returns:
        max_total_tokens: 可分配的总 token 数
    """
    # 如果手动指定了 token 数，直接返回
    if max_total_tokens_override is not None and max_total_tokens_override > 0:
        logger.info(
            f"Using manual max_total_tokens override: {max_total_tokens_override}"
        )
        return max_total_tokens_override

    # 获取 GPU 显存统计
    memory_stats = get_gpu_memory_stats(device)
    available_memory = int(memory_stats.free_bytes * memory_ratio)

    # 计算每个 token 的显存占用
    memory_per_token = estimate_kv_cache_memory_per_token(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # 计算 token 数
    num_tokens = available_memory // memory_per_token

    logger.info(
        f"KV Cache allocation: "
        f"free_memory={memory_stats.free_gb:.2f}GB, "
        f"usable={available_memory / GB:.2f}GB (ratio={memory_ratio}), "
        f"memory_per_token={memory_per_token}B, "
        f"max_total_tokens={num_tokens}"
    )

    return max(1, num_tokens)


class MemoryBudgetManager:
    """
    显存预算管理器

    负责:
    1. 计算和管理 KV cache 可用容量 (num_pages / max_total_tokens)
    2. 跟踪已分配和可用的显存
    3. 提供预算检查接口

    Attributes:
        num_pages: 总页面数
        max_total_tokens: 最大 token 数
        page_size: 每页 token 数
        bytes_per_token: 每个 token 的显存占用
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        page_size: int = 256,
        gpu_memory_utilization: float = 0.9,
        max_total_tokens: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        device: str = "cuda",
    ):
        """
        初始化显存预算管理器

        Args:
            num_layers: 模型层数
            num_kv_heads: KV head 数量
            head_dim: head 维度
            dtype: 数据类型
            page_size: 每页 token 数
            gpu_memory_utilization: GPU 显存利用率
            max_total_tokens: 手动指定的最大 token 数（用于 KV cache 容量）
            max_num_batched_tokens: 单次推理的最大 token 数（用于 prefill 调度）
            device: GPU 设备
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.page_size = page_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.device = device

        # 计算每个 token 的显存占用
        self.bytes_per_token = estimate_kv_cache_memory_per_token(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        # 兼容旧属性名
        self.memory_per_token = self.bytes_per_token

        # 计算最大 token 数
        self.max_total_tokens = _determine_num_tokens(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            memory_ratio=gpu_memory_utilization,
            max_total_tokens_override=max_total_tokens,
            device=device,
        )

        # 计算页面数
        self.num_pages = (self.max_total_tokens + page_size - 1) // page_size

        # 设置单次推理的最大 token 数
        if max_num_batched_tokens is None:
            # 默认值：使用 max_total_tokens
            self.max_num_batched_tokens = self.max_total_tokens
        else:
            self.max_num_batched_tokens = max_num_batched_tokens

        # 当前分配的 token 数（由 token allocator 管理，这里只是记录）
        self._allocated_tokens = 0

        logger.info(
            f"MemoryBudgetManager initialized: "
            f"max_total_tokens={self.max_total_tokens}, "
            f"num_pages={self.num_pages}, "
            f"page_size={self.page_size}, "
            f"max_num_batched_tokens={self.max_num_batched_tokens}, "
            f"bytes_per_token={self.bytes_per_token}B"
        )

    @property
    def available_tokens(self) -> int:
        """返回可用的 token 数"""
        return self.max_total_tokens - self._allocated_tokens

    @property
    def kv_cache_memory_bytes(self) -> int:
        """返回 KV cache 总显存占用"""
        return self.max_total_tokens * self.bytes_per_token

    def can_allocate(self, num_tokens: int) -> bool:
        """检查是否可以分配指定数量的 tokens"""
        return self.available_tokens >= num_tokens

    def allocate(self, num_tokens: int) -> bool:
        """分配 tokens (仅用于跟踪)"""
        if not self.can_allocate(num_tokens):
            return False
        self._allocated_tokens += num_tokens
        return True

    def free(self, num_tokens: int):
        """释放 tokens (仅用于跟踪)"""
        self._allocated_tokens = max(0, self._allocated_tokens - num_tokens)

    def estimate_tokens_from_memory(self, memory_bytes: int) -> int:
        """从内存大小估算可容纳的 token 数"""
        return memory_bytes // self.bytes_per_token

    def estimate_memory_from_tokens(self, num_tokens: int) -> int:
        """从 token 数量估算所需内存"""
        return num_tokens * self.bytes_per_token

    def get_batch_token_budget(self) -> int:
        """获取单次推理的 token 预算"""
        return self.max_num_batched_tokens

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "max_total_tokens": self.max_total_tokens,
            "num_pages": self.num_pages,
            "page_size": self.page_size,
            "allocated_tokens": self._allocated_tokens,
            "available_tokens": self.available_tokens,
            "bytes_per_token": self.bytes_per_token,
            "memory_per_token": self.bytes_per_token,
            "kv_cache_memory_gb": self.kv_cache_memory_bytes / GB,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }


class TokenBudgetAdder:
    """
    Token 预算加法器

    用于在调度过程中跟踪当前 batch 的 token 数量，
    确保不超过 token_budget 限制。支持分别跟踪 extend 和 decode tokens。
    """

    def __init__(
        self,
        token_budget: int,
        max_batch_size: int = 256,
        reserved_decode_tokens: int = 0,
    ):
        """
        Args:
            token_budget: 单次推理的最大 token 数
            max_batch_size: 最大 batch 大小
            reserved_decode_tokens: 为 decode 预留的 token 数
        """
        self.token_budget = token_budget
        self.max_batch_size = max_batch_size
        self.reserved_decode_tokens = reserved_decode_tokens

        # 跟踪状态
        self.current_extend_tokens = 0
        self.current_decode_tokens = 0
        self.extend_reqs: List[Any] = []
        self.decode_reqs: List[Any] = []

    @property
    def total_tokens(self) -> int:
        """当前已添加的总 token 数"""
        return self.current_extend_tokens + self.current_decode_tokens

    @property
    def total_reqs(self) -> int:
        """当前已添加的总请求数"""
        return len(self.extend_reqs) + len(self.decode_reqs)

    def can_add_extend(self, num_tokens: int) -> bool:
        """检查是否可以添加 extend tokens"""
        if self.total_reqs >= self.max_batch_size:
            return False
        # 可用于 extend 的预算 = token_budget - reserved_decode_tokens - 已用
        available = (
            self.token_budget - self.reserved_decode_tokens - self.current_extend_tokens
        )
        return num_tokens <= available

    def add_extend(self, req: Any, num_tokens: int) -> bool:
        """
        添加 extend 请求

        Returns:
            True 如果添加成功
        """
        if not self.can_add_extend(num_tokens):
            return False
        self.extend_reqs.append(req)
        self.current_extend_tokens += num_tokens
        return True

    def can_add_decode(self) -> bool:
        """检查是否可以添加 decode 请求"""
        if self.total_reqs >= self.max_batch_size:
            return False
        # decode 每个请求消耗 1 token
        return self.total_tokens + 1 <= self.token_budget

    def add_decode(self, req: Any) -> bool:
        """
        添加 decode 请求

        Returns:
            True 如果添加成功
        """
        if not self.can_add_decode():
            return False
        self.decode_reqs.append(req)
        self.current_decode_tokens += 1
        return True

    def remaining(self) -> int:
        """返回剩余可添加的 token 数"""
        return self.token_budget - self.total_tokens

    def reset(self):
        """重置计数器"""
        self.current_extend_tokens = 0
        self.current_decode_tokens = 0
        self.extend_reqs = []
        self.decode_reqs = []

    def get_batch_summary(self) -> dict:
        """获取 batch 摘要"""
        return {
            "extend_reqs": len(self.extend_reqs),
            "decode_reqs": len(self.decode_reqs),
            "extend_tokens": self.current_extend_tokens,
            "decode_tokens": self.current_decode_tokens,
            "total_tokens": self.total_tokens,
        }


class PrefillAdder:
    """
    Prefill 预算控制器

    支持 chunked prefill，可以将大的 prefill 请求分成多个 chunk 处理。
    同时跟踪 token 预算和请求数量。
    """

    def __init__(
        self,
        prefill_budget: Optional[int] = None,
        reserved_size: int = 0,
        kv_cache_mgr: Any = None,
        max_batch_size: int = 256,
        chunk_size: int = 8192,
    ):
        """
        Args:
            prefill_budget: Prefill token 预算 (旧 API)
            reserved_size: 为 decode 预留的 token 数
            kv_cache_mgr: KV cache 管理器
            max_batch_size: 最大 batch 大小
            chunk_size: 分块大小
            max_num_batched_tokens: 单次推理的最大 token 数 (新 API)
            max_num_seqs: 单次推理的最大请求数 (新 API)
            max_extend_tokens: 单个请求的最大 extend token 数
        """
        self.prefill_budget = prefill_budget
        self.reserved_size = reserved_size
        self.kv_cache_mgr = kv_cache_mgr
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size

        # 跟踪状态
        self.current_prefill_tokens = 0
        self.prefill_reqs: List[Any] = []
        self.chunked_reqs: List[Any] = []

    @property
    def total_tokens(self) -> int:
        """当前已添加的总 token 数"""
        return self.current_prefill_tokens + self.current_decode_tokens

    @property
    def available_prefill_budget(self) -> int:
        """可用的 prefill token 预算"""
        return self.prefill_budget - self.current_prefill_tokens

    def try_add_prefill(
        self,
        req: Any,
        chunk_size: Optional[int] = None,
    ) -> Optional[Tuple[Any, int, bool]]:
        """
        尝试添加 prefill 请求

        Args:
            req: 请求对象，需要有 extend_input_len 属性
            chunk_size: 可选的分块大小

        Returns:
            (req, actual_extend_len, is_chunked) 或 None
        """
        # 检查 batch size 限制
        if (
            len(self.prefill_reqs) + self.reserved_size + len(self.chunked_reqs)
            >= self.max_batch_size
        ):
            return None

        extend_len = getattr(req, "extend_input_len", 0)

        # 检查 KV cache 可用空间
        available = self.kv_cache_mgr.available_tokens()
        if extend_len > available:
            return None

        # 检查预算
        if self.current_prefill_tokens >= self.prefill_budget:
            return None

        # 计算可处理的长度
        remaining_budget = self.available_prefill_budget
        actual_chunk_size = chunk_size or self.chunk_size
        actual_len = min(extend_len, remaining_budget, actual_chunk_size)

        if actual_len <= 0:
            return None

        is_chunked = actual_len < extend_len

        if is_chunked:
            self.chunked_reqs.append(req)
            req.is_chunked = True
        else:
            self.prefill_reqs.append(req)

        self.current_prefill_tokens += actual_len

        return (req, actual_len, is_chunked)


def compute_max_total_tokens(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    gpu_memory_utilization: float = 0.9,
    device: str = "cuda",
) -> int:
    """
    便捷函数：计算 KV cache 可容纳的最大 token 数

    Args:
        num_layers: 模型层数
        num_kv_heads: KV head 数量
        head_dim: head 维度
        dtype: 数据类型
        gpu_memory_utilization: GPU 显存利用率
        device: GPU 设备

    Returns:
        max_total_tokens: 可分配的总 token 数
    """
    return _determine_num_tokens(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        memory_ratio=gpu_memory_utilization,
        device=device,
    )
