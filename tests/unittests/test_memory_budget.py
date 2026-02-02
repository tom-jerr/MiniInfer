"""
测试 Memory Budget 和 PrefillAdder
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from miniinfer.kvcache.memory_budget import (
    MemoryBudgetManager,
    PrefillAdder,
    TokenBudgetAdder,
    MemoryStats,
    get_gpu_memory_stats,
    estimate_kv_cache_memory_per_token,
)


class TestEstimateKVCacheMemory:
    """测试 KV cache 内存估算"""

    def test_estimate_kv_cache_memory_per_token_fp16(self):
        """测试 FP16 下每个 token 的内存估算"""
        # 假设 Qwen2-7B 的配置
        num_layers = 32
        num_kv_heads = 8  # GQA
        head_dim = 128
        dtype = torch.float16

        bytes_per_token = estimate_kv_cache_memory_per_token(
            num_layers, num_kv_heads, head_dim, dtype
        )

        # 每层: K + V = 2 * num_kv_heads * head_dim * 2 bytes (fp16)
        # = 2 * 8 * 128 * 2 = 4096 bytes per layer
        # 总共: 4096 * 32 = 131072 bytes = 128 KB per token
        expected = 2 * num_kv_heads * head_dim * 2 * num_layers
        assert bytes_per_token == expected
        assert bytes_per_token == 131072  # 128 KB

    def test_estimate_kv_cache_memory_per_token_bf16(self):
        """测试 BF16 下每个 token 的内存估算"""
        num_layers = 24
        num_kv_heads = 4
        head_dim = 64
        dtype = torch.bfloat16

        bytes_per_token = estimate_kv_cache_memory_per_token(
            num_layers, num_kv_heads, head_dim, dtype
        )

        # 每层: K + V = 2 * 4 * 64 * 2 = 1024 bytes
        # 总共: 1024 * 24 = 24576 bytes
        expected = 2 * num_kv_heads * head_dim * 2 * num_layers
        assert bytes_per_token == expected


class TestMemoryBudgetManager:
    """测试 MemoryBudgetManager"""

    def test_init(self):
        """测试初始化"""
        mgr = MemoryBudgetManager(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            gpu_memory_utilization=0.9,
            max_total_tokens=20480,
            max_num_batched_tokens=8192,
        )

        assert mgr.bytes_per_token == 131072  # 128 KB
        assert mgr.max_num_batched_tokens == 8192
        assert mgr.max_total_tokens == 20480

    def test_estimate_tokens_from_memory(self):
        """测试从内存估算 token 数量"""
        mgr = MemoryBudgetManager(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            gpu_memory_utilization=0.9,
            max_total_tokens=20480,
        )

        # 1 GB 可以存储多少 token
        memory_bytes = 1024 * 1024 * 1024  # 1 GB
        tokens = mgr.estimate_tokens_from_memory(memory_bytes)

        # 1 GB / 128 KB = 8192 tokens
        expected = memory_bytes // 131072
        assert tokens == expected

    def test_estimate_memory_from_tokens(self):
        """测试从 token 数量估算内存"""
        mgr = MemoryBudgetManager(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            gpu_memory_utilization=0.9,
            max_total_tokens=20480,
        )

        # 1000 tokens 需要多少内存
        memory = mgr.estimate_memory_from_tokens(1000)
        expected = 1000 * 131072
        assert memory == expected

    def test_get_batch_token_budget(self):
        """测试获取 batch token 预算"""
        mgr = MemoryBudgetManager(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_total_tokens=20480,
            max_num_batched_tokens=8192,
        )

        assert mgr.get_batch_token_budget() == 8192

    def test_get_batch_token_budget_default(self):
        """测试默认 batch token 预算"""
        mgr = MemoryBudgetManager(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            max_total_tokens=20480,
            max_num_batched_tokens=None,  # 使用默认值
        )

        assert mgr.get_batch_token_budget() == 20480


class TestTokenBudgetAdder:
    """测试 TokenBudgetAdder"""

    def test_init_and_reset(self):
        """测试初始化和重置"""
        adder = TokenBudgetAdder(
            token_budget=8192,
            max_batch_size=256,
            reserved_decode_tokens=256,
        )

        assert adder.token_budget == 8192
        assert adder.max_batch_size == 256
        assert adder.total_tokens == 0
        assert adder.total_reqs == 0

        # 模拟添加一些请求
        adder.current_extend_tokens = 100
        adder.current_decode_tokens = 10

        # 重置
        adder.reset()
        assert adder.current_extend_tokens == 0
        assert adder.current_decode_tokens == 0

    def test_can_add_extend_within_budget(self):
        """测试在预算内添加 extend 请求"""
        adder = TokenBudgetAdder(
            token_budget=8192,
            max_batch_size=256,
            reserved_decode_tokens=256,
        )

        # 可用预算 = 8192 - 256 = 7936 for extend
        assert adder.can_add_extend(1000) is True
        assert adder.can_add_extend(7936) is True
        assert adder.can_add_extend(7937) is False  # 超出预算

    def test_add_extend_accumulates(self):
        """测试累加 extend token"""
        adder = TokenBudgetAdder(
            token_budget=8192,
            max_batch_size=256,
            reserved_decode_tokens=256,
        )

        class MockReq:
            pass

        req1, req2 = MockReq(), MockReq()

        assert adder.add_extend(req1, 1000) is True
        assert adder.current_extend_tokens == 1000
        assert len(adder.extend_reqs) == 1

        assert adder.add_extend(req2, 2000) is True
        assert adder.current_extend_tokens == 3000
        assert len(adder.extend_reqs) == 2

    def test_can_add_decode(self):
        """测试添加 decode 请求"""
        adder = TokenBudgetAdder(
            token_budget=100,
            max_batch_size=10,
        )

        class MockReq:
            pass

        # 可以添加 decode
        for i in range(10):
            assert adder.can_add_decode() is True
            assert adder.add_decode(MockReq()) is True

        # 达到 max_batch_size 限制
        assert adder.can_add_decode() is False

    def test_batch_size_limit(self):
        """测试 batch size 限制"""
        adder = TokenBudgetAdder(
            token_budget=10000,
            max_batch_size=3,
        )

        class MockReq:
            pass

        # 添加 3 个请求
        for i in range(3):
            assert adder.add_extend(MockReq(), 100) is True

        # 第 4 个应该失败（超出 batch size）
        assert adder.can_add_extend(100) is False
        assert adder.add_extend(MockReq(), 100) is False

    def test_mixed_extend_decode(self):
        """测试混合 extend 和 decode"""
        adder = TokenBudgetAdder(
            token_budget=1000,
            max_batch_size=100,
            reserved_decode_tokens=100,
        )

        class MockReq:
            pass

        # 添加 extend
        assert adder.add_extend(MockReq(), 500) is True
        assert adder.total_tokens == 500

        # 添加 decode
        for i in range(50):
            assert adder.add_decode(MockReq()) is True

        assert adder.total_tokens == 550
        assert adder.current_extend_tokens == 500
        assert adder.current_decode_tokens == 50

    def test_get_batch_summary(self):
        """测试获取 batch 摘要"""
        adder = TokenBudgetAdder(
            token_budget=1000,
            max_batch_size=100,
        )

        class MockReq:
            pass

        adder.add_extend(MockReq(), 200)
        adder.add_extend(MockReq(), 300)
        adder.add_decode(MockReq())
        adder.add_decode(MockReq())

        summary = adder.get_batch_summary()
        assert summary["extend_reqs"] == 2
        assert summary["decode_reqs"] == 2
        assert summary["extend_tokens"] == 500
        assert summary["decode_tokens"] == 2
        assert summary["total_tokens"] == 502


class TestMemoryStats:
    """测试 MemoryStats 数据类"""

    def test_memory_stats_properties(self):
        """测试属性计算"""
        GB = 1024 * 1024 * 1024
        stats = MemoryStats(
            total_bytes=8 * GB,
            allocated_bytes=2 * GB,
            reserved_bytes=3 * GB,
            free_bytes=5 * GB,
        )

        assert stats.total_gb == 8.0
        assert stats.allocated_gb == 2.0
        assert stats.free_gb == 5.0


class TestPrefillAdder:
    """测试 PrefillAdder - Chunked Prefill 的核心组件"""

    def _create_mock_kv_cache_mgr(self, available_tokens: int = 10000):
        """创建 mock KVCacheManager"""
        mgr = Mock()
        mgr.available_tokens.return_value = available_tokens
        return mgr

    def _create_mock_req(
        self, extend_input_len: int = 100, cache_protected_len: int = 0
    ):
        """创建 mock Req"""
        req = Mock()
        req.extend_input_len = extend_input_len
        req.cache_protected_len = cache_protected_len
        req.total_input_len = extend_input_len + cache_protected_len
        req.is_chunked = False
        return req

    def test_init(self):
        """测试初始化"""
        mgr = self._create_mock_kv_cache_mgr()
        adder = PrefillAdder(
            prefill_budget=8192,
            reserved_size=100,
            kv_cache_mgr=mgr,
            max_batch_size=256,
        )

        assert adder.prefill_budget == 8192
        assert adder.reserved_size == 100
        assert adder.total_tokens == 0
        assert adder.available_prefill_budget == 8192

    def test_add_decode_priority(self):
        """测试 decode 请求优先"""
        mgr = self._create_mock_kv_cache_mgr()
        adder = PrefillAdder(
            prefill_budget=1000,
            reserved_size=100,  # 100 个 decode 请求
            kv_cache_mgr=mgr,
        )

        # 添加 decode 请求
        for i in range(10):
            req = self._create_mock_req()
            assert adder.add_decode(req) is True

        assert len(adder.decode_reqs) == 10
        assert adder.current_decode_tokens == 10

    def test_try_add_prefill_complete(self):
        """测试完整 prefill（不需要 chunk）"""
        mgr = self._create_mock_kv_cache_mgr(available_tokens=10000)
        adder = PrefillAdder(
            prefill_budget=8192,
            reserved_size=0,
            kv_cache_mgr=mgr,
        )

        req = self._create_mock_req(extend_input_len=500)
        result = adder.try_add_prefill(req)

        assert result is not None
        added_req, extend_len, is_chunked = result
        assert added_req == req
        assert extend_len == 500
        assert is_chunked is False
        assert len(adder.prefill_reqs) == 1
        assert adder.current_prefill_tokens == 500

    def test_try_add_prefill_chunked(self):
        """测试 chunked prefill"""
        mgr = self._create_mock_kv_cache_mgr(available_tokens=10000)
        adder = PrefillAdder(
            prefill_budget=1000,  # 只有 1000 的预算
            reserved_size=0,
            kv_cache_mgr=mgr,
        )

        # 请求需要 5000 tokens，超过预算
        req = self._create_mock_req(extend_input_len=5000)
        result = adder.try_add_prefill(req, chunk_size=1000)

        assert result is not None
        added_req, actual_len, is_chunked = result
        assert added_req == req
        assert actual_len == 1000  # 只处理 1000
        assert is_chunked is True
        assert len(adder.chunked_reqs) == 1

    def test_try_add_prefill_kv_cache_full(self):
        """测试 KV cache 满时无法添加"""
        mgr = self._create_mock_kv_cache_mgr(available_tokens=100)  # 只有 100
        adder = PrefillAdder(
            prefill_budget=8192,
            reserved_size=0,
            kv_cache_mgr=mgr,
        )

        # 请求需要 500 tokens，超过可用 KV cache
        req = self._create_mock_req(extend_input_len=500)
        result = adder.try_add_prefill(req)

        assert result is None

    def test_prefill_budget_respects_decode(self):
        """测试 prefill 预算尊重 decode 预留"""
        mgr = self._create_mock_kv_cache_mgr()
        adder = PrefillAdder(
            prefill_budget=1000,
            reserved_size=200,  # 200 个 decode token 预留
            kv_cache_mgr=mgr,
        )

        # 先添加一些 decode
        for i in range(50):
            adder.add_decode(self._create_mock_req())

        # 剩余 prefill 预算应该仍然是 1000
        assert adder.available_prefill_budget == 1000

        # 添加 prefill
        req = self._create_mock_req(extend_input_len=800)
        result = adder.try_add_prefill(req)
        assert result is not None

        # 剩余预算
        assert adder.available_prefill_budget == 200

    def test_batch_size_limit(self):
        """测试 batch size 限制"""
        mgr = self._create_mock_kv_cache_mgr()
        adder = PrefillAdder(
            prefill_budget=10000,
            reserved_size=0,
            kv_cache_mgr=mgr,
            max_batch_size=5,
        )

        # 添加 5 个请求
        for i in range(5):
            req = self._create_mock_req(extend_input_len=100)
            result = adder.try_add_prefill(req)
            assert result is not None

        # 第 6 个应该失败
        req = self._create_mock_req(extend_input_len=100)
        result = adder.try_add_prefill(req)
        assert result is None

    def test_get_batch_summary(self):
        """测试获取 batch 摘要"""
        mgr = self._create_mock_kv_cache_mgr()
        adder = PrefillAdder(
            prefill_budget=8192,
            reserved_size=0,
            kv_cache_mgr=mgr,
        )

        # 添加一些请求
        for i in range(3):
            adder.add_decode(self._create_mock_req())

        req = self._create_mock_req(extend_input_len=500)
        adder.try_add_prefill(req)

        summary = adder.get_batch_summary()
        assert summary["prefill_reqs"] == 1
        assert summary["decode_reqs"] == 3
        assert summary["prefill_tokens"] == 500
        assert summary["decode_tokens"] == 3
        assert summary["total_tokens"] == 503

    def test_reset(self):
        """测试重置状态"""
        mgr = self._create_mock_kv_cache_mgr()
        adder = PrefillAdder(
            prefill_budget=8192,
            reserved_size=0,
            kv_cache_mgr=mgr,
        )

        # 添加一些请求
        adder.add_decode(self._create_mock_req())
        adder.try_add_prefill(self._create_mock_req(500))

        # 重置
        adder.reset()

        assert adder.current_prefill_tokens == 0
        assert adder.current_decode_tokens == 0
        assert len(adder.prefill_reqs) == 0
        assert len(adder.decode_reqs) == 0
        assert len(adder.chunked_reqs) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUMemoryStats:
    """测试 GPU 显存统计（需要 CUDA）"""

    def test_get_gpu_memory_stats(self):
        """测试获取 GPU 显存统计"""
        stats = get_gpu_memory_stats()

        assert stats.total_bytes > 0
        assert stats.allocated_bytes >= 0
        assert stats.reserved_bytes >= 0
        assert stats.free_bytes >= 0

    def test_memory_stats_after_allocation(self):
        """测试分配后的显存统计变化"""
        stats_before = get_gpu_memory_stats()

        # 分配一些显存
        tensor = torch.zeros(1024, 1024, device="cuda")

        stats_after = get_gpu_memory_stats()

        # 已分配的应该增加
        assert stats_after.allocated_bytes >= stats_before.allocated_bytes

        # 清理
        del tensor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
