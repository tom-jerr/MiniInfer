"""
KV Cache Manager 单元测试

测试内容:
1. TokenAllocator: 分配/释放 KV indices
2. RequestPool: 请求槽位管理和映射
3. MHAKVCacheStorage: KV buffer 读写
4. RadixCache: 前缀匹配和缓存
5. KVCacheManager: 完整流程
"""

import pytest
import torch
import sys
from kvcache.memory_pool import TokenAllocator, RequestPool, MHAKVCacheStorage
from kvcache.radix_cache import RadixCache
from kvcache.kv_cache_manager import KVCacheManager


# ============== 测试配置 ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on device: {DEVICE}")


# ============== TokenAllocator 测试 ==============
class TestTokenAllocator:
    """测试 Token 分配器"""

    def test_basic_alloc_free(self):
        """基本分配和释放"""
        allocator = TokenAllocator(size=100, device=DEVICE)

        # 初始可用空间
        assert allocator.available_size() == 100

        # 分配 10 个 tokens
        indices = allocator.alloc(10)
        assert indices is not None
        assert len(indices) == 10
        assert allocator.available_size() == 90

        # indices 应该从 1 开始 (0 保留)
        assert indices[0].item() >= 1

        # 释放
        allocator.free(indices)
        assert allocator.available_size() == 100

    def test_alloc_exceed_capacity(self):
        """超出容量的分配"""
        allocator = TokenAllocator(size=10, device=DEVICE)

        # 分配超过容量
        indices = allocator.alloc(20)
        assert indices is None
        assert allocator.available_size() == 10  # 没有变化

    def test_sequential_alloc(self):
        """连续分配"""
        allocator = TokenAllocator(size=100, device=DEVICE)

        indices1 = allocator.alloc(30)
        indices2 = allocator.alloc(30)
        indices3 = allocator.alloc(30)

        assert allocator.available_size() == 10

        # 所有 indices 应该不重叠
        all_indices = torch.cat([indices1, indices2, indices3])
        unique_indices = torch.unique(all_indices)
        assert len(unique_indices) == 90

    def test_free_and_reuse(self):
        """释放后重新分配"""
        allocator = TokenAllocator(size=50, device=DEVICE)

        # 分配全部
        indices1 = allocator.alloc(50)
        assert allocator.available_size() == 0

        # 释放一半
        allocator.free(indices1[:25])
        assert allocator.available_size() == 25

        # 重新分配
        indices2 = allocator.alloc(25)
        assert indices2 is not None
        assert allocator.available_size() == 0


# ============== RequestPool 测试 ==============
class TestRequestPool:
    """测试请求池"""

    def test_basic_alloc_free(self):
        """基本请求槽位分配"""
        pool = RequestPool(max_requests=10, max_context_len=1024, device=DEVICE)

        # 分配请求槽位
        slots = pool.alloc(3)
        assert slots is not None
        assert len(slots) == 3

        # 释放
        pool.free(slots)

    def test_write_read_mapping(self):
        """测试 token 到 KV 位置的映射"""
        pool = RequestPool(max_requests=10, max_context_len=1024, device=DEVICE)

        # 分配一个请求
        slots = pool.alloc(1)
        req_idx = slots[0]

        # 写入映射: token 0-9 -> kv_loc 100-109
        kv_indices = torch.arange(100, 110, dtype=torch.int64, device=DEVICE)
        pool.write(req_idx, slice(0, 10), kv_indices)

        # 读取映射
        read_indices = pool.read(req_idx, slice(0, 10))
        assert torch.equal(read_indices.to(torch.int64), kv_indices.to(torch.int64))

    def test_exceed_max_requests(self):
        """超出最大请求数"""
        pool = RequestPool(max_requests=5, max_context_len=1024, device=DEVICE)

        # 分配全部
        slots = pool.alloc(5)
        assert slots is not None

        # 再次分配应该失败
        more_slots = pool.alloc(1)
        assert more_slots is None


# ============== MHAKVCacheStorage 测试 ==============
class TestMHAKVCacheStorage:
    """测试 KV Cache 物理存储"""

    def test_basic_get_set(self):
        """基本读写"""
        storage = MHAKVCacheStorage(
            size=100,
            page_size=1,
            num_layers=2,
            num_heads=8,
            head_dim=64,
            dtype=torch.float16,
            device=DEVICE,
        )

        # 获取 buffer
        k_buffer, v_buffer = storage.get_kv_buffer(layer_id=0)
        assert k_buffer.shape == (101, 8, 64)  # size + page_size
        assert v_buffer.shape == (101, 8, 64)

        # 写入数据
        loc = torch.tensor([1, 2, 3], device=DEVICE)
        cache_k = torch.randn(3, 8, 64, dtype=torch.float16, device=DEVICE)
        cache_v = torch.randn(3, 8, 64, dtype=torch.float16, device=DEVICE)

        storage.set_kv_buffer(layer_id=0, loc=loc, cache_k=cache_k, cache_v=cache_v)

        # 验证写入
        k_buffer, v_buffer = storage.get_kv_buffer(layer_id=0)
        assert torch.allclose(k_buffer[1:4], cache_k)
        assert torch.allclose(v_buffer[1:4], cache_v)

    def test_multiple_layers(self):
        """多层存储"""
        num_layers = 4
        storage = MHAKVCacheStorage(
            size=50,
            page_size=1,
            num_layers=num_layers,
            num_heads=4,
            head_dim=32,
            dtype=torch.float16,
            device=DEVICE,
        )

        # 每层写入不同数据
        for layer_id in range(num_layers):
            loc = torch.tensor([1], device=DEVICE)
            cache_k = torch.full(
                (1, 4, 32), layer_id, dtype=torch.float16, device=DEVICE
            )
            cache_v = torch.full(
                (1, 4, 32), layer_id + 10, dtype=torch.float16, device=DEVICE
            )
            storage.set_kv_buffer(layer_id, loc, cache_k, cache_v)

        # 验证每层数据正确
        for layer_id in range(num_layers):
            k_buffer, v_buffer = storage.get_kv_buffer(layer_id)
            assert k_buffer[1, 0, 0].item() == layer_id
            assert v_buffer[1, 0, 0].item() == layer_id + 10


# ============== RadixCache 测试 (简化版) ==============
class TestRadixCache:
    """测试前缀缓存"""

    def test_basic_insert_match(self):
        """基本插入和匹配"""
        allocator = TokenAllocator(size=1000, device=DEVICE)
        cache = RadixCache(token_allocator=allocator, page_size=1)

        # 插入一个序列
        key = [1, 2, 3, 4, 5]
        value = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64, device=DEVICE)
        cache.insert(key, value)

        # 完全匹配
        matched_value, node = cache.match_prefix(key)
        assert len(matched_value) == 5

        # 前缀匹配
        longer_key = [1, 2, 3, 4, 5, 6, 7]
        matched_value, node = cache.match_prefix(longer_key)
        assert len(matched_value) == 5  # 只匹配前 5 个

    def test_no_match(self):
        """无匹配"""
        allocator = TokenAllocator(size=1000, device=DEVICE)
        cache = RadixCache(token_allocator=allocator, page_size=1)

        # 插入
        cache.insert(
            [1, 2, 3], torch.tensor([10, 20, 30], dtype=torch.int64, device=DEVICE)
        )

        # 不同前缀，无匹配
        matched_value, node = cache.match_prefix([4, 5, 6])
        assert len(matched_value) == 0

    def test_partial_match(self):
        """部分匹配"""
        allocator = TokenAllocator(size=1000, device=DEVICE)
        cache = RadixCache(token_allocator=allocator, page_size=1)

        # 插入
        cache.insert(
            [1, 2, 3, 4],
            torch.tensor([10, 20, 30, 40], dtype=torch.int64, device=DEVICE),
        )

        # 部分匹配
        matched_value, node = cache.match_prefix([1, 2, 5, 6])
        assert len(matched_value) == 2  # 只有 [1, 2] 匹配


# ============== KVCacheManager 集成测试 ==============
class TestKVCacheManager:
    """测试完整的 KV Cache 管理器"""

    @pytest.fixture
    def manager(self):
        """创建测试用的 manager"""
        return KVCacheManager(
            size=1000,
            max_requests=10,
            max_context_len=512,
            num_layers=2,
            num_heads=4,
            head_dim=32,
            dtype=torch.float16,
            device=DEVICE,
            enable_prefix_cache=True,
            page_size=1,
        )

    def test_initialization(self, manager):
        """测试初始化"""
        assert manager.available_tokens() == 1000
        assert manager.can_allocate(100)
        assert manager.can_allocate(1000)
        assert not manager.can_allocate(1001)

    def test_request_lifecycle(self, manager):
        """测试请求生命周期: 分配 -> 使用 -> 释放"""
        # 1. 分配请求槽位
        req_idx = manager.alloc_request()
        assert req_idx is not None

        # 2. 为请求分配 KV cache
        token_ids = [1, 2, 3, 4, 5]
        kv_indices, num_cached = manager.alloc_for_request(
            req_idx=req_idx,
            token_ids=token_ids,
            num_new_tokens=5,
        )

        assert kv_indices is not None
        assert len(kv_indices) == 5
        assert num_cached == 0  # 第一次没有缓存
        assert manager.available_tokens() == 995

        # 3. 释放请求
        manager.release_request(
            req_idx=req_idx,
            token_ids=token_ids,
            num_tokens=5,
            cache_to_radix=True,
        )

        # KV cache 被缓存到 radix tree，不会立即释放
        # 请求槽位被释放

    def test_prefix_cache_hit(self, manager):
        """测试前缀缓存命中"""
        # 第一个请求
        req_idx1 = manager.alloc_request()
        token_ids1 = [1, 2, 3, 4, 5]
        kv_indices1, num_cached1 = manager.alloc_for_request(
            req_idx=req_idx1,
            token_ids=token_ids1,
            num_new_tokens=5,
        )
        assert num_cached1 == 0

        # 释放请求 (缓存到 radix tree)
        manager.release_request(
            req_idx=req_idx1,
            token_ids=token_ids1,
            num_tokens=5,
            cache_to_radix=True,
        )

        # 第二个请求，相同前缀
        req_idx2 = manager.alloc_request()
        token_ids2 = [1, 2, 3, 4, 5, 6, 7]  # 前 5 个相同
        kv_indices2, num_cached2 = manager.alloc_for_request(
            req_idx=req_idx2,
            token_ids=token_ids2,
            num_new_tokens=7,
        )

        # 应该命中 5 个 token 的缓存
        assert num_cached2 == 5
        # 只需要分配 2 个新 token
        assert len(kv_indices2) == 2

    def test_multiple_requests(self, manager):
        """测试多个并发请求"""
        # 分配多个请求
        req_indices = []
        for i in range(5):
            req_idx = manager.alloc_request()
            assert req_idx is not None
            req_indices.append(req_idx)

        # 为每个请求分配 KV cache
        for req_idx in req_indices:
            token_ids = list(range(req_idx * 10, req_idx * 10 + 20))
            kv_indices, _ = manager.alloc_for_request(
                req_idx=req_idx,
                token_ids=token_ids,
                num_new_tokens=20,
            )
            assert kv_indices is not None

        # 验证总共分配了 5 * 20 = 100 tokens
        assert manager.available_tokens() == 900

    def test_kv_buffer_operations(self, manager):
        """测试 KV buffer 读写"""
        # 分配请求
        req_idx = manager.alloc_request()
        token_ids = [1, 2, 3]
        kv_indices, _ = manager.alloc_for_request(
            req_idx=req_idx,
            token_ids=token_ids,
            num_new_tokens=3,
        )

        # 写入 KV cache
        cache_k = torch.randn(3, 4, 32, dtype=torch.float16, device=DEVICE)
        cache_v = torch.randn(3, 4, 32, dtype=torch.float16, device=DEVICE)

        manager.set_kv_buffer(
            layer_id=0,
            loc=kv_indices,
            cache_k=cache_k,
            cache_v=cache_v,
        )

        # 读取 KV buffer
        k_buffer, v_buffer = manager.get_kv_buffer(layer_id=0)

        # 验证写入正确
        for i, loc in enumerate(kv_indices):
            assert torch.allclose(k_buffer[loc], cache_k[i])
            assert torch.allclose(v_buffer[loc], cache_v[i])

    def test_stats(self, manager):
        """测试统计信息"""
        stats = manager.get_stats()

        assert "size" in stats
        assert "available_tokens" in stats
        assert "used_tokens" in stats
        assert "utilization" in stats

        assert stats["size"] == 1000
        assert stats["available_tokens"] == 1000
        assert stats["utilization"] == 0.0


# ============== 运行测试 ==============
if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    print("=" * 60)
    print("KV Cache Manager Unit Tests")
    print("=" * 60)

    # 使用 pytest 运行
    pytest.main([__file__, "-v", "--tb=short"])
