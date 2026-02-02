from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
import torch


class IKVCacheStorage(ABC):
    """KV Cache 物理存储接口"""

    @abstractmethod
    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的 K, V buffer"""
        pass

    @abstractmethod
    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        """将 K, V 写入指定位置"""
        pass

    @abstractmethod
    def get_kv_size_bytes(self) -> int:
        """获取 KV cache 占用的字节数"""
        pass


class ITokenAllocator(ABC):
    """Token 到 KV Cache 位置的分配器接口"""

    @abstractmethod
    def alloc(self, num_tokens: int) -> Optional[torch.Tensor]:
        """分配 KV cache 位置，返回位置索引"""
        pass

    @abstractmethod
    def alloc_pages_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
    ) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def alloc_pages_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def free(self, indices: torch.Tensor):
        """释放 KV cache 位置"""
        pass

    @abstractmethod
    def available_size(self) -> int:
        """返回可用槽位数"""
        pass


class IRequestPool(ABC):
    """请求池接口"""

    @abstractmethod
    def req_to_token_pool(self) -> Any:
        pass

    @abstractmethod
    def alloc(self, num_reqs: int) -> Optional[List[int]]:
        """分配请求槽位"""
        pass

    @abstractmethod
    def free(self, indices: List[int]):
        """释放请求槽位"""
        pass

    @abstractmethod
    def write(self, req_idx: int, token_range: slice, kv_indices: torch.Tensor):
        """写入 token 到 KV cache 的映射"""
        pass

    @abstractmethod
    def read(self, req_idx: int, token_range: slice) -> torch.Tensor:
        """读取映射"""
        pass


class IPrefixCache(ABC):
    """前缀缓存接口"""

    @abstractmethod
    def match_prefix(self, key: List[int]) -> Tuple[torch.Tensor, Any]:
        """匹配前缀，返回 (KV索引, 节点)"""
        pass

    @abstractmethod
    def insert(self, key: List[int], value=None) -> int:
        """插入新的前缀，返回新前缀长度"""
        pass

    @abstractmethod
    def evict(self, num_tokens: int):
        """驱逐缓存"""
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any):
        """增加节点引用"""
        pass

    @abstractmethod
    def dec_lock_ref(self, node: Any):
        """减少节点引用"""
        pass
