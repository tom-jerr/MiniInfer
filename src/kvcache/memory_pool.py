import torch
from enum import Enum
import numpy as np
from typing import Optional, Union, List, Tuple, Any
import logging
from .interface import IKVCacheStorage, ITokenAllocator, IRequestPool


logger = logging.getLogger(__name__)
GB = 1024 * 1024 * 1024


class MHAKVCacheStorage(IKVCacheStorage):

    def __init__(
        self,
        size: int,
        page_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.size = size
        self.page_size = page_size
        self.num_layers = num_layers
        self.device = device

        # 分配 K, V buffer
        # Shape: [max_tokens + 1, num_heads, head_dim]  (位置0保留)
        self.k_buffer = [
            torch.zeros(
                (size + self.page_size, num_heads, head_dim), dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]
        self.v_buffer = [
            torch.zeros(
                (size + self.page_size, num_heads, head_dim), dtype=dtype, device=device
            )
            for _ in range(num_layers)
        ]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v

    def get_kv_size_bytes(self) -> int:
        """获取 KV cache 占用的字节数"""
        total_bytes = 0
        for k, v in zip(self.k_buffer, self.v_buffer):
            total_bytes += k.numel() * k.element_size()
            total_bytes += v.numel() * v.element_size()
        return total_bytes


class TokenAllocator(ITokenAllocator):

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        # 位置 0 保留，从 1 开始
        self.free_slots = torch.arange(1, size + 1, dtype=torch.int64, device=device)

    def alloc(self, num_tokens: int) -> Optional[torch.Tensor]:
        if num_tokens > len(self.free_slots):
            return None
        allocated = self.free_slots[:num_tokens]
        self.free_slots = self.free_slots[num_tokens:]
        return allocated

    def free(self, indices: torch.Tensor):
        if indices.numel() > 0:
            self.free_slots = torch.cat([self.free_slots, indices])

    def available_size(self) -> int:
        return len(self.free_slots)


class RequestPool(IRequestPool):
    """

    Args:
        max_requests: max running batch size
        max_context_len: from model config, max context length
    """

    def __init__(self, max_requests: int, max_context_len: int, device: str):
        self.max_requests = max_requests
        self.max_context_len = max_context_len
        self.device = device

        # req_to_token[req_idx, token_pos] = kv_loc
        self.req_to_token = torch.zeros(
            (max_requests, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(max_requests))

    def alloc(self, num_reqs: int) -> Optional[List[int]]:
        if num_reqs > len(self.free_slots):
            return None
        allocated = self.free_slots[:num_reqs]
        self.free_slots = self.free_slots[num_reqs:]
        return allocated

    def free(self, indices: List[int]):
        self.free_slots.extend(indices)

    # TODO(lzy): use cuda kernel to accelerate
    def write(self, req_idx: int, token_range: slice, kv_indices: torch.Tensor):
        self.req_to_token[req_idx, token_range] = kv_indices.to(torch.int32)

    def read(self, req_idx: int, token_range: slice) -> torch.Tensor:
        return self.req_to_token[req_idx, token_range].to(torch.int64)
