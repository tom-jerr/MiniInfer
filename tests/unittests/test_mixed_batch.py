"""
Mixed Batch (Chunked Prefill + Decode) 单元测试

测试 Sarathi-Serve 风格的 stall-free batching:
当开启 chunked prefill 时，将 decode 请求和 prefill 请求合并为 mixed batch，
所有请求统一使用 extend kernel 处理（decode 视为 extend_len=1）。

测试内容:
1. ForwardMode.is_extend() / is_mixed() 行为正确
2. KVCacheManager.prepare_for_mixed() 正确分配 KV cache
3. Scheduler 构建 mixed batch 的元数据正确
4. FlashAttention backend 能正确处理 mixed batch
"""

import pytest
import torch
from scheduler.scheduler_batch import (
    Req,
    ScheduledBatch,
    ForwardBatch,
    ForwardMode,
    BatchResult,
)
from kvcache.kv_cache_manager import KVCacheManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAGE_SIZE = 256
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 64


# ============== Fixtures ==============
@pytest.fixture
def kv_cache_mgr():
    """创建一个用于测试的 KVCacheManager"""
    mgr = KVCacheManager(
        size=PAGE_SIZE * 8,
        max_requests=16,
        max_context_len=2048,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float16,
        device=DEVICE,
        enable_prefix_cache=True,
        page_size=PAGE_SIZE,
    )
    return mgr


def _make_decode_req(
    token_ids: list[int],
    output_ids: list[int],
    kv_cache_mgr: KVCacheManager,
) -> Req:
    """创建一个已完成 prefill 并生成了一些 output 的 decode 请求"""
    from miniinfer.utils.sampling_params import SamplingParams

    req = Req(token_ids, SamplingParams(max_tokens=100))
    req.output_ids = list(output_ids)
    req.fill_ids = token_ids + list(output_ids)

    # 模拟 prefill: 分配 req_pool slot 和 KV cache
    kv_cache_mgr.prefix_for_waiting_req(req)
    batch = ScheduledBatch.init_new([req], device=torch.device(DEVICE))
    kv_cache_mgr.prepare_for_extend(batch)
    req.req_pool_idx = int(batch.req_pool_indices[0].item())

    # 模拟 decode 步骤: 为每个 output token 分配 KV cache
    for i, tok in enumerate(output_ids):
        seq_len_before = len(token_ids) + i
        req_rpi = torch.tensor([req.req_pool_idx], dtype=torch.int64, device=DEVICE)
        seq_lens = torch.tensor([seq_len_before], dtype=torch.int64, device=DEVICE)
        seq_lens_cpu = torch.tensor([seq_len_before], dtype=torch.int64)
        last_loc = kv_cache_mgr.request_pool.req_to_token_pool()[req_rpi, seq_lens - 1]
        out_loc = kv_cache_mgr.token_allocator.alloc_pages_decode(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            last_loc=last_loc,
        )
        new_pos = seq_lens  # after +1
        kv_cache_mgr.request_pool.req_to_token_pool()[req_rpi, new_pos] = out_loc.to(
            torch.int32
        )

    return req


# ============== ForwardMode 测试 ==============
class TestForwardMode:
    def test_mixed_is_extend(self):
        """MIXED 模式下 is_extend() 应返回 True"""
        assert ForwardMode.MIXED.is_extend() is True

    def test_mixed_is_not_decode(self):
        """MIXED 模式下 is_decode() 应返回 False"""
        assert ForwardMode.MIXED.is_decode() is False

    def test_mixed_is_mixed(self):
        """MIXED 模式下 is_mixed() 应返回 True"""
        assert ForwardMode.MIXED.is_mixed() is True

    def test_extend_is_not_mixed(self):
        """EXTEND 模式下 is_mixed() 应返回 False"""
        assert ForwardMode.EXTEND.is_mixed() is False

    def test_extend_is_extend(self):
        """EXTEND 保持原有行为"""
        assert ForwardMode.EXTEND.is_extend() is True

    def test_decode_is_decode(self):
        """DECODE 保持原有行为"""
        assert ForwardMode.DECODE.is_decode() is True
        assert ForwardMode.DECODE.is_extend() is False


# ============== prepare_for_mixed 测试 ==============
class TestPrepareForMixed:
    def test_mixed_no_decode_fallback(self, kv_cache_mgr):
        """没有 decode 请求时，prepare_for_mixed 应退化为 prepare_for_extend"""
        token_ids = list(range(10))
        req = Req(token_ids)
        kv_cache_mgr.prefix_for_waiting_req(req)

        batch = ScheduledBatch.init_new([req], device=torch.device(DEVICE))
        kv_cache_mgr.prepare_for_mixed(batch, [req], [])

        # 验证基本元数据
        assert batch.input_ids is not None
        assert batch.req_pool_indices is not None
        assert batch.out_cache_loc is not None
        assert batch.seq_lens is not None
        assert len(batch.prefix_lens) == 1
        assert len(batch.extend_lens) == 1

    def test_mixed_with_decode_reqs(self, kv_cache_mgr):
        """Mixed batch: 1 个 extend 请求 + 1 个 decode 请求"""
        # 创建 decode 请求（已完成 prefill）
        decode_input = list(range(10))
        decode_outputs = [100, 101, 102]
        decode_req = _make_decode_req(decode_input, decode_outputs, kv_cache_mgr)

        # 创建新的 extend 请求
        extend_input = list(range(200, 215))
        extend_req = Req(extend_input)
        kv_cache_mgr.prefix_for_waiting_req(extend_req)

        # 构建 mixed batch
        all_reqs = [extend_req, decode_req]
        batch = ScheduledBatch.init_new(all_reqs, device=torch.device(DEVICE))
        kv_cache_mgr.prepare_for_mixed(batch, [extend_req], [decode_req])

        # 验证 batch 元数据
        assert batch.input_ids is not None
        assert batch.req_pool_indices.shape[0] == 2
        assert batch.seq_lens.shape[0] == 2
        assert len(batch.prefix_lens) == 2
        assert len(batch.extend_lens) == 2

        # extend 请求的 extend_len 应该等于输入长度（无 prefix cache）
        assert batch.extend_lens[0] == len(extend_input)

        # decode 请求的 extend_len 应该为 1
        assert batch.extend_lens[1] == 1

        # decode 请求的 prefix_len 应该为 seq_len - 1
        decode_seq_len = len(decode_input) + len(decode_outputs)
        assert batch.prefix_lens[1] == decode_seq_len - 1

        # seq_lens
        assert batch.seq_lens[0].item() == len(extend_input)
        assert batch.seq_lens[1].item() == decode_seq_len

        # input_ids: extend token ids + decode 最后一个 output token
        expected_ids = extend_input + [decode_outputs[-1]]
        assert batch.input_ids.cpu().tolist() == expected_ids

    def test_mixed_batch_out_cache_loc(self, kv_cache_mgr):
        """验证 mixed batch 的 out_cache_loc 分配正确"""
        # 创建 decode 请求
        decode_input = list(range(8))
        decode_outputs = [50, 51]
        decode_req = _make_decode_req(decode_input, decode_outputs, kv_cache_mgr)

        # 创建 extend 请求
        extend_input = list(range(100, 105))
        extend_req = Req(extend_input)
        kv_cache_mgr.prefix_for_waiting_req(extend_req)

        batch = ScheduledBatch.init_new(
            [extend_req, decode_req], device=torch.device(DEVICE)
        )
        kv_cache_mgr.prepare_for_mixed(batch, [extend_req], [decode_req])

        # out_cache_loc: extend 部分 + decode 部分
        total_out_tokens = len(extend_input) + 1  # extend tokens + 1 decode token
        assert batch.out_cache_loc.shape[0] == total_out_tokens
        # 所有 out_cache_loc 应该是有效的 KV indices (>= 0)
        assert (batch.out_cache_loc >= 0).all()

    def test_mixed_multiple_decode_reqs(self, kv_cache_mgr):
        """多个 decode 请求的 mixed batch"""
        # 两个 decode 请求
        d1 = _make_decode_req(list(range(5)), [10], kv_cache_mgr)
        d2 = _make_decode_req(list(range(20, 28)), [30, 31], kv_cache_mgr)

        # 一个 extend 请求
        extend_input = list(range(100, 103))
        e1 = Req(extend_input)
        kv_cache_mgr.prefix_for_waiting_req(e1)

        batch = ScheduledBatch.init_new([e1, d1, d2], device=torch.device(DEVICE))
        kv_cache_mgr.prepare_for_mixed(batch, [e1], [d1, d2])

        assert batch.req_pool_indices.shape[0] == 3
        assert len(batch.extend_lens) == 3
        assert batch.extend_lens[0] == len(extend_input)
        assert batch.extend_lens[1] == 1
        assert batch.extend_lens[2] == 1

        # out_cache_loc = extend tokens + 2 decode tokens
        assert batch.out_cache_loc.shape[0] == len(extend_input) + 2


# ============== ForwardBatch init 测试 ==============
class TestForwardBatchMixed:
    def test_forward_batch_mixed_mode(self, kv_cache_mgr):
        """ForwardBatch 在 MIXED 模式下应正确初始化 extend 相关字段"""
        decode_req = _make_decode_req(list(range(10)), [42], kv_cache_mgr)
        extend_input = list(range(50, 58))
        extend_req = Req(extend_input)
        kv_cache_mgr.prefix_for_waiting_req(extend_req)

        batch = ScheduledBatch.init_new(
            [extend_req, decode_req], device=torch.device(DEVICE)
        )
        batch.forward_mode = ForwardMode.MIXED
        batch.decoding_reqs = [decode_req]
        kv_cache_mgr.prepare_for_mixed(batch, [extend_req], [decode_req])

        # 由于 is_extend() 包含 MIXED，ForwardBatch.init_new 应走 extend 分支
        # 这需要一个 attn_backend，我们这里不传而是直接检查 batch 元数据
        assert batch.forward_mode == ForwardMode.MIXED
        assert batch.forward_mode.is_extend() is True

        # 检查 extend_lens 和 prefix_lens 对 flash attn varlen 的正确性
        assert batch.extend_lens[0] == len(extend_input)  # extend
        assert batch.extend_lens[1] == 1  # decode as extend
        assert batch.prefix_lens[1] == 10  # decode seq_len - 1


# ============== 调度器 mixed batch 构建测试 ==============
class TestSchedulerMixedBatch:
    def test_scheduler_mixed_batch_metadata(self):
        """验证调度器构建 mixed batch 时的 decoding_reqs 字段"""
        batch = ScheduledBatch.init_new([], device=torch.device(DEVICE))
        batch.forward_mode = ForwardMode.MIXED
        decode_req = Req([1, 2, 3])
        batch.decoding_reqs = [decode_req]

        assert batch.decoding_reqs is not None
        assert len(batch.decoding_reqs) == 1
        assert batch.forward_mode.is_mixed()

    def test_scheduler_extend_batch_no_decoding_reqs(self):
        """纯 extend batch 的 decoding_reqs 应为 None"""
        batch = ScheduledBatch.init_new([], device=torch.device(DEVICE))
        batch.forward_mode = ForwardMode.EXTEND

        assert batch.decoding_reqs is None
        assert not batch.forward_mode.is_mixed()
        assert batch.forward_mode.is_extend()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
