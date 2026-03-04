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
from engine.scheduler_batch import Req, ScheduledBatch
from kvcache.memory_pool import (
  TokenAllocator,
  RequestPool,
  MHAKVCacheStorage,
  PagedTokenAllocator,
)
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
      cache_k = torch.full((1, 4, 32), layer_id, dtype=torch.float16, device=DEVICE)
      cache_v = torch.full((1, 4, 32), layer_id + 10, dtype=torch.float16, device=DEVICE)
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
    cache.insert([1, 2, 3], torch.tensor([10, 20, 30], dtype=torch.int64, device=DEVICE))

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
PAGE_SIZE = 256


@pytest.fixture
def paged_manager_factory():
  """Provide a KVCacheManager factory using Triton kernels (requires CUDA)."""
  if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton allocator tests")

  def _factory(*, enable_prefix_cache: bool = True, size: int = PAGE_SIZE * 4) -> KVCacheManager:
    return KVCacheManager(
      size=size,
      max_requests=8,
      max_context_len=1024,
      num_layers=1,
      num_heads=1,
      head_dim=8,
      dtype=torch.float32,
      device="cuda",
      enable_prefix_cache=enable_prefix_cache,
      page_size=PAGE_SIZE,
    )

  return _factory


@pytest.fixture
def paged_allocator():
  if not torch.cuda.is_available():
    pytest.skip("CUDA required for Triton allocator tests")
  return PagedTokenAllocator(size=PAGE_SIZE * 4, page_size=PAGE_SIZE, device="cuda")


def _print_pages(label: str, out: torch.Tensor, page_size: int = PAGE_SIZE):
  """Print page-wise view with zero padding for unused slots."""
  page_ids = torch.unique(out // page_size).tolist()
  print(f"[{label}] pages_used={len(page_ids)}, page_ids={page_ids}")
  for pid in page_ids:
    mask = (out // page_size) == pid
    locs = out[mask]
    page_view = torch.zeros(page_size, dtype=torch.int64)
    offsets = (locs % page_size).to(torch.int64)
    page_view[offsets] = locs.to(torch.int64)
    print(f"  page {pid}: {page_view.tolist()}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton allocator")
class TestPagedTokenAllocator:
  def test_extend_across_pages(self, paged_allocator: PagedTokenAllocator):
    prefix_lens = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor([PAGE_SIZE + 44], dtype=torch.int64, device=DEVICE)
    last_loc = torch.tensor([-1], dtype=torch.int64, device=DEVICE)
    out = paged_allocator.alloc_pages_extend(
      prefix_lens, prefix_lens.cpu(), seq_lens, seq_lens.cpu(), last_loc, int(seq_lens.item())
    )

    assert out.numel() == PAGE_SIZE + 44
    page_ids = out // PAGE_SIZE
    assert torch.unique(page_ids).numel() == 2
    assert (page_ids == page_ids[0]).sum().item() == PAGE_SIZE
    assert (page_ids == page_ids[-1]).sum().item() == 44
    assert paged_allocator.available_size() == 2 * PAGE_SIZE  # two pages consumed
    _print_pages("alloc_extend_two_pages", out.cpu())

  def test_extend_fills_partial_page_before_new(self, paged_allocator: PagedTokenAllocator):
    # simulate pages 1 & 2 already in use
    paged_allocator.free_pages = paged_allocator.free_pages[2:]

    prefix_lens = torch.tensor([PAGE_SIZE + 10], dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor([PAGE_SIZE + 20], dtype=torch.int64, device=DEVICE)
    last_loc = torch.tensor(
      [PAGE_SIZE * 2 + 10 - 1], dtype=torch.int64, device=DEVICE
    )  # page 2, offset 9
    out = paged_allocator.alloc_pages_extend(
      prefix_lens, prefix_lens.cpu(), seq_lens, seq_lens.cpu(), last_loc, 10
    )

    assert torch.unique(out // PAGE_SIZE).tolist() == [2]
    assert (out // PAGE_SIZE == 2).sum().item() == 10
    assert paged_allocator.available_size() == 2 * PAGE_SIZE  # no new page consumed
    _print_pages("alloc_extend_partial_page", out.cpu())

  def test_extend_from_existing_partial_page_print(self, paged_allocator: PagedTokenAllocator):
    # Simulate prefix occupying page 1 (remove it from free list)
    paged_allocator.free_pages = paged_allocator.free_pages[1:]

    prefix_len = PAGE_SIZE - 10  # leave 10 slots unused on page 1
    seq_len = prefix_len + 30  # fill remaining 10 + 20 into a new page
    prefix_lens = torch.tensor([prefix_len], dtype=torch.int64, device=DEVICE)
    seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=DEVICE)
    last_loc = torch.tensor([PAGE_SIZE + prefix_len - 1], dtype=torch.int64, device=DEVICE)

    out = paged_allocator.alloc_pages_extend(
      prefix_lens,
      prefix_lens.cpu(),
      seq_lens,
      seq_lens.cpu(),
      last_loc,
      seq_len - prefix_len,
    )

    page_ids = torch.unique(out // PAGE_SIZE).tolist()
    assert len(page_ids) == 2
    assert (out // PAGE_SIZE == page_ids[0]).sum().item() == 10  # tail of page 1
    assert (out // PAGE_SIZE == page_ids[1]).sum().item() == 20  # new page portion
    # free pages left: started with pages 2,3,4. One new page consumed -> 2 pages free
    assert paged_allocator.available_size() == 2 * PAGE_SIZE
    _print_pages("alloc_extend_partial_existing_page", out.cpu())

  def test_decode_new_page_when_last_loc_at_page_end(self, paged_allocator: PagedTokenAllocator):
    # consume page 1 so decoding will use next free page
    paged_allocator.free_pages = paged_allocator.free_pages[1:]

    seq_lens = torch.tensor([PAGE_SIZE + 1], dtype=torch.int64, device=DEVICE)
    last_loc = torch.tensor([PAGE_SIZE * 2 - 1], dtype=torch.int64, device=DEVICE)  # end of page 1
    out = paged_allocator.alloc_pages_decode(seq_lens, seq_lens.cpu(), last_loc)

    assert out.numel() == 1
    assert out[0].item() % PAGE_SIZE == 0  # new page start
    assert paged_allocator.available_size() == 2 * PAGE_SIZE  # one page consumed from three
    _print_pages("alloc_decode_new_page", out.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton allocator")
class TestPagedKVCacheManager:
  def test_cross_page_extend_and_release(self, paged_manager_factory):
    manager = paged_manager_factory(enable_prefix_cache=False, size=PAGE_SIZE * 2)
    req = Req(list(range(PAGE_SIZE + 44)))

    manager.prefix_for_waiting_req(req)
    batch = ScheduledBatch.init_new([req], device=torch.device(DEVICE))
    manager.prepare_for_extend(batch)

    req.req_pool_idx = int(batch.req_pool_indices[0].item())
    req.req_id = req.req_pool_idx

    out_loc = batch.out_cache_loc.cpu()
    assert out_loc.numel() == PAGE_SIZE + 44
    assert out_loc[0].item() % PAGE_SIZE == 0
    assert out_loc[PAGE_SIZE - 1].item() % PAGE_SIZE == PAGE_SIZE - 1
    assert out_loc[PAGE_SIZE].item() % PAGE_SIZE == 0

    page_ids = out_loc // PAGE_SIZE
    assert torch.unique(page_ids[:PAGE_SIZE]).numel() == 1
    assert torch.unique(page_ids[PAGE_SIZE:]).numel() == 1
    assert page_ids[0].item() != page_ids[-1].item()
    first_page_tokens = (page_ids == page_ids[0]).sum().item()
    second_page_tokens = (page_ids == page_ids[-1]).sum().item()
    print(
      f"[cross_page] total_tokens={out_loc.numel()}, pages_used={torch.unique(page_ids).numel()}, "
      f"first_page_tokens={first_page_tokens}, second_page_tokens={second_page_tokens}, "
      f"last_loc={out_loc[-1].item()}"
    )
    _print_pages("kv_cross_page", out_loc)
    assert second_page_tokens == 44  # only partial page write for the tail

    mapped = manager.request_pool.read(req.req_pool_idx, slice(0, len(req.fill_ids))).cpu()
    assert torch.equal(mapped.to(torch.int64), out_loc.to(torch.int64))

    assert manager.available_tokens() == 0

    manager.release_request(req, is_insert=False)
    assert manager.available_tokens() == manager.size

  def test_prefix_cache_reuse_across_pages(self, paged_manager_factory):
    manager = paged_manager_factory(enable_prefix_cache=True, size=PAGE_SIZE * 4)

    req1_tokens = list(range(PAGE_SIZE * 2))  # two full pages
    req1 = Req(req1_tokens)
    manager.prefix_for_waiting_req(req1)
    batch1 = ScheduledBatch.init_new([req1], device=torch.device(DEVICE))
    manager.prepare_for_extend(batch1)

    req1.req_pool_idx = int(batch1.req_pool_indices[0].item())
    req1.req_id = req1.req_pool_idx
    kv_indices1 = manager.request_pool.read(req1.req_pool_idx, slice(0, len(req1.fill_ids)))
    manager.prefix_cache.insert(req1.fill_ids, kv_indices1)
    print(
      f"[prefix_cache_hit] req1_prefix_len={len(req1_tokens)}, "
      f"req1_pages_used={torch.unique(kv_indices1 // PAGE_SIZE).numel()}"
    )
    _print_pages("kv_prefix_req1_pages", kv_indices1.cpu())

    req2_tokens = req1_tokens + list(range(1000, 1008))
    req2 = Req(req2_tokens)
    manager.prefix_for_waiting_req(req2)
    assert req2.cache_protected_len == len(req1_tokens)
    assert req2.prefix_indices.shape[0] == len(req1_tokens)
    assert torch.equal(req2.prefix_indices.cpu(), kv_indices1.cpu())

    batch2 = ScheduledBatch.init_new([req2], device=torch.device(DEVICE))
    manager.prepare_for_extend(batch2)

    req2.req_pool_idx = int(batch2.req_pool_indices[0].item())
    req2.req_id = req2.req_pool_idx

    out_loc = batch2.out_cache_loc.cpu()
    assert out_loc.numel() == len(req2_tokens) - len(req1_tokens)
    assert torch.unique(out_loc // PAGE_SIZE).tolist() == [3]
    tail_page_tokens = out_loc.numel()
    print(
      f"[prefix_cache_hit] extend_tokens={out_loc.numel()}, new_page={int(out_loc[0] // PAGE_SIZE)}, "
      f"tail_page_tokens={tail_page_tokens}"
    )
    _print_pages("kv_prefix_extend_pages", out_loc.cpu())
    assert tail_page_tokens == 8  # extend writes into a fresh page partially

    mapped2 = manager.request_pool.read(req2.req_pool_idx, slice(0, len(req2.fill_ids))).cpu()
    assert torch.equal(
      mapped2[: len(req1_tokens)].to(torch.int64), kv_indices1.to(device="cpu", dtype=torch.int64)
    )
    assert torch.equal(mapped2[len(req1_tokens) :].to(torch.int64), out_loc.to(torch.int64))

    assert manager.available_tokens() == PAGE_SIZE

  def test_partial_prefix_cache_reuse_across_pages(self, paged_manager_factory):
    manager = paged_manager_factory(enable_prefix_cache=True, size=PAGE_SIZE * 4)

    req1_tokens = list(range(PAGE_SIZE + 100))  # two full pages
    req1 = Req(req1_tokens)
    manager.prefix_for_waiting_req(req1)
    batch1 = ScheduledBatch.init_new([req1], device=torch.device(DEVICE))
    manager.prepare_for_extend(batch1)

    req1.req_pool_idx = int(batch1.req_pool_indices[0].item())
    req1.req_id = req1.req_pool_idx
    kv_indices1 = manager.request_pool.read(req1.req_pool_idx, slice(0, len(req1.fill_ids)))
    manager.prefix_cache.insert(req1.fill_ids, kv_indices1)
    print(
      f"[prefix_cache_hit] req1_prefix_len={len(req1_tokens)}, "
      f"req1_pages_used={torch.unique(kv_indices1 // PAGE_SIZE).numel()}"
    )
    # _print_pages("kv_prefix_req1_pages", kv_indices1.cpu())

    req2_tokens = req1_tokens[:PAGE_SIZE] + list(range(1000, 1008))
    req2 = Req(req2_tokens)
    manager.prefix_for_waiting_req(req2)
    assert req2.cache_protected_len == PAGE_SIZE  # only first page hits
    assert req2.prefix_indices.shape[0] == PAGE_SIZE
    assert torch.equal(req2.prefix_indices.cpu()[:PAGE_SIZE], kv_indices1.cpu()[:PAGE_SIZE])

    batch2 = ScheduledBatch.init_new([req2], device=torch.device(DEVICE))
    print("req2 prefix_indices:", req2.prefix_indices)
    manager.prepare_for_extend(batch2)
    print("after prepare_for_extend, req2 prefix_indices:", req2.prefix_indices)

    req2.req_pool_idx = int(batch2.req_pool_indices[0].item())
    req2.req_id = req2.req_pool_idx

    out_loc = batch2.out_cache_loc.cpu()
    assert out_loc.numel() == len(req2_tokens) - PAGE_SIZE
    assert torch.unique(out_loc // PAGE_SIZE).tolist() == [3]
    tail_page_tokens = out_loc.numel()
    print(
      f"[prefix_cache_hit] extend_tokens={out_loc.numel()}, new_page={int(out_loc[0] // PAGE_SIZE)}, "
      f"tail_page_tokens={tail_page_tokens}"
    )
    # _print_pages("kv_prefix_extend_pages", out_loc.cpu())
    assert tail_page_tokens == 8  # extend writes into a fresh page partially

    # _print_pages("all_mapped_req2", manager.request_pool.read(req2.req_pool_idx, slice(0, len(req2.fill_ids))).cpu())
    mapped2 = manager.request_pool.read(req2.req_pool_idx, slice(0, len(req2.fill_ids))).cpu()
    assert torch.equal(
      mapped2[:PAGE_SIZE].to(torch.int64),
      kv_indices1[:PAGE_SIZE].to(device="cpu", dtype=torch.int64),
    )
    assert torch.equal(mapped2[PAGE_SIZE:].to(torch.int64), out_loc.to(torch.int64))

  def test_release_request_unlocks_radix_and_preserves_pages(self, paged_manager_factory):
    manager = paged_manager_factory(enable_prefix_cache=True, size=PAGE_SIZE * 3)

    # seed radix cache with a one-page request
    req_base = Req(list(range(PAGE_SIZE)))
    manager.prefix_for_waiting_req(req_base)
    batch_base = ScheduledBatch.init_new([req_base], device=torch.device(DEVICE))
    manager.prepare_for_extend(batch_base)
    req_base.req_pool_idx = int(batch_base.req_pool_indices[0].item())
    req_base.req_id = req_base.req_pool_idx
    manager.release_request(req_base, is_insert=True)

    # one page occupied by radix cache, two pages free
    assert manager.available_tokens() == PAGE_SIZE * 2

    # second request reuses the cached page and extends by 16 tokens
    req_hit = Req(list(range(PAGE_SIZE)) + list(range(1000, 1016)))
    manager.prefix_for_waiting_req(req_hit)
    assert req_hit.cache_protected_len == PAGE_SIZE

    # lock the matched radix node to make protected_size non-zero
    manager.prefix_cache.inc_lock_ref(req_hit.last_node)
    assert manager.prefix_cache.protected_size() == PAGE_SIZE

    batch_hit = ScheduledBatch.init_new([req_hit], device=torch.device(DEVICE))
    manager.prepare_for_extend(batch_hit)
    req_hit.req_pool_idx = int(batch_hit.req_pool_indices[0].item())
    req_hit.req_id = req_hit.req_pool_idx

    manager.release_request(req_hit, is_insert=True)

    # radix cache keeps two pages cached, leaving exactly one free page
    assert manager.available_tokens() == PAGE_SIZE
    assert manager.prefix_cache.protected_size() == 0
    assert req_hit.last_node.lock_ref == 0
