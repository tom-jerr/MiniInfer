import torch
from enum import Enum
import numpy as np
from typing import Optional, Union, List, Tuple, Any
from miniinfer.utils import get_logger
from .interface import IKVCacheStorage, ITokenAllocator, IRequestPool
import triton
import triton.language as tl

logger = get_logger(__name__)
GB = 1024 * 1024 * 1024


def next_power_of_2(n: int):
  return 1 << (n - 1).bit_length() if n > 0 else 1


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
      torch.zeros((size + self.page_size, num_heads, head_dim), dtype=dtype, device=device)
      for _ in range(num_layers)
    ]
    self.v_buffer = [
      torch.zeros((size + self.page_size, num_heads, head_dim), dtype=dtype, device=device)
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

  def alloc_pages_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
  ) -> Optional[torch.Tensor]:
    raise NotImplementedError()

  def alloc_pages_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
  ) -> Optional[torch.Tensor]:
    raise NotImplementedError()


@triton.jit
def alloc_extend_kernel(
  pre_lens_ptr,
  seq_lens_ptr,
  last_loc_ptr,
  free_page_ptr,
  out_indices,
  bs_upper: tl.constexpr,
  page_size: tl.constexpr,
  max_num_extend_tokens: tl.constexpr,
):
  pid = tl.program_id(0)

  load_offset = tl.arange(0, bs_upper)
  seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
  pre_lens = tl.load(pre_lens_ptr + load_offset, mask=load_offset <= pid)
  extend_lens = seq_lens - pre_lens

  seq_len = tl.load(seq_lens_ptr + pid)
  pre_len = tl.load(pre_lens_ptr + pid)
  extend_len = seq_len - pre_len

  sum_extend_lens = tl.sum(extend_lens)
  output_start_loc = sum_extend_lens - extend_len

  num_pages_after = (seq_lens + page_size - 1) // page_size
  num_pages_before = (pre_lens + page_size - 1) // page_size
  num_new_pages = num_pages_after - num_pages_before

  num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
    pre_len + page_size - 1
  ) // page_size
  sum_num_new_pages = tl.sum(num_new_pages)
  new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

  # Part 1: fill the old partial page
  last_loc = tl.load(last_loc_ptr + pid)
  num_part1 = min(seq_len, (pre_len + page_size - 1) // page_size * page_size) - pre_len
  offset_one_page = tl.arange(0, page_size)
  tl.store(
    out_indices + output_start_loc + offset_one_page,
    last_loc + 1 + offset_one_page,
    mask=offset_one_page < num_part1,
  )
  if pre_len + num_part1 == seq_len:
    return

  # Part 2: fill the new full pages
  num_part2 = seq_len // page_size * page_size - (pre_len + page_size - 1) // page_size * page_size

  offset_many_page = tl.arange(0, max_num_extend_tokens)
  page_start = tl.load(
    free_page_ptr + new_page_start_loc + offset_many_page // page_size,
    mask=offset_many_page < num_part2,
  )
  tl.store(
    out_indices + output_start_loc + num_part1 + offset_many_page,
    page_start * page_size + offset_many_page % page_size,
    mask=offset_many_page < num_part2,
  )
  if pre_len + num_part1 + num_part2 == seq_len:
    return

  # Part 3: fill the new partial page
  num_part3 = seq_len - seq_len // page_size * page_size
  start_loc = tl.load(free_page_ptr + new_page_start_loc + num_page_start_loc_self - 1)
  tl.store(
    out_indices + output_start_loc + num_part1 + num_part2 + offset_one_page,
    start_loc * page_size + offset_one_page,
    mask=offset_one_page < num_part3,
  )


@triton.jit
def alloc_decode_kernel(
  seq_lens_ptr,
  last_loc_ptr,
  free_page_ptr,
  out_indices,
  bs_upper: tl.constexpr,
  page_size: tl.constexpr,
):
  pid = tl.program_id(0)

  load_offset = tl.arange(0, bs_upper)
  seq_lens = tl.load(seq_lens_ptr + load_offset, mask=load_offset <= pid)
  pre_lens = tl.where(load_offset <= pid, seq_lens - 1, seq_lens)

  seq_len = tl.load(seq_lens_ptr + pid)
  pre_len = seq_len - 1

  num_pages_after = (seq_lens + page_size - 1) // page_size
  num_pages_before = (pre_lens + page_size - 1) // page_size
  num_new_pages = num_pages_after - num_pages_before

  num_page_start_loc_self = (seq_len + page_size - 1) // page_size - (
    pre_len + page_size - 1
  ) // page_size
  sum_num_new_pages = tl.sum(num_new_pages)
  new_page_start_loc = sum_num_new_pages - num_page_start_loc_self

  if num_page_start_loc_self == 0:
    last_loc = tl.load(last_loc_ptr + pid)
    tl.store(out_indices + pid, last_loc + 1)
  else:
    page = tl.load(free_page_ptr + new_page_start_loc)
    tl.store(out_indices + pid, page * page_size)


def get_num_new_pages(
  seq_lens: torch.Tensor,
  page_size: int,
  prefix_lens: Optional[torch.Tensor] = None,
  decode: bool = False,
) -> torch.Tensor:
  """
  Get the number of new pages for the given prefix and sequence lengths.
  We use cpu tensors to avoid blocking kernel launch.
  """
  cpu_device = torch.device("cpu")
  assert seq_lens.device == cpu_device

  if prefix_lens is None or decode:
    # NOTE: Special case for handling decode, which prefix lens is `seq_lens - 1`.
    # 如果 seq_lens % page_size == 1，说明这个新 Token 刚好是新的一页的第1个 Token。
    # 必须分配 new page
    assert decode
    return (seq_lens % page_size == 1).int().sum().item()

  assert prefix_lens.device == cpu_device
  num_pages_after = (seq_lens + page_size - 1) // page_size
  num_pages_before = (prefix_lens + page_size - 1) // page_size
  num_new_pages = num_pages_after - num_pages_before
  sum_num_new_pages = torch.sum(num_new_pages).to(torch.int64)
  return sum_num_new_pages.item()


class PagedTokenAllocator(ITokenAllocator):
  def __init__(self, size: int, page_size: int, device: str):
    self.size = size
    self.page_size = page_size
    self.device = device
    self.num_pages = size // page_size
    self.seen_max_num_extend_tokens_next_power_of_2 = 1
    # keep allocator in a usable default state
    self.debug_mode = False
    self.need_sort = False
    self.clear()

  def alloc(self, need_size: int):
    # page-aligned allocation, returning contiguous indices of pages
    assert need_size % self.page_size == 0, "The allocation size should be page-aligned"

    num_pages = need_size // self.page_size
    if num_pages > len(self.free_pages):
      return None

    out_pages = self.free_pages[:num_pages]
    self.free_pages = self.free_pages[num_pages:]

    out_indices = (
      out_pages[:, None] * self.page_size + torch.arange(self.page_size, device=self.device)
    ).reshape(-1)

    return out_indices

  def alloc_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
  ):
    if self.debug_mode:
      assert torch.all((last_loc + 1) % self.page_size == prefix_lens % self.page_size)

    # 先检查是否有足够的 free pages，避免 kernel 从无效位置读取
    num_new_pages = get_num_new_pages(
      seq_lens=seq_lens_cpu,
      page_size=self.page_size,
      prefix_lens=prefix_lens_cpu,
    )
    if num_new_pages > len(self.free_pages):
      return None

    self.seen_max_num_extend_tokens_next_power_of_2 = max(
      self.seen_max_num_extend_tokens_next_power_of_2,
      next_power_of_2(extend_num_tokens),
    )

    bs = len(prefix_lens)
    out_indices = torch.empty((extend_num_tokens,), dtype=torch.int64, device=self.device)

    alloc_extend_kernel[(bs,)](
      prefix_lens,
      seq_lens,
      last_loc,
      self.free_pages,
      out_indices,
      next_power_of_2(bs),
      self.page_size,
      self.seen_max_num_extend_tokens_next_power_of_2,
    )

    assert len(torch.unique(out_indices)) == len(out_indices)

    # print("num_new_pages:", num_new_pages)
    # print("out_indices:", out_indices)

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices

  def alloc_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
  ):
    if self.debug_mode:
      assert torch.all((last_loc + 2) % self.page_size == seq_lens % self.page_size)

    bs = len(seq_lens)
    if self.need_sort and bs > len(self.free_pages):
      self.merge_and_sort_free()

    # 检查是否有足够的页 - 必须在内核执行之前检查
    num_new_pages = get_num_new_pages(
      seq_lens=seq_lens_cpu,
      page_size=self.page_size,
      decode=True,
    )
    if num_new_pages > len(self.free_pages):
      return None

    out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
    alloc_decode_kernel[(bs,)](
      seq_lens,
      last_loc,
      self.free_pages,
      out_indices,
      next_power_of_2(bs),
      self.page_size,
    )

    if self.debug_mode:
      assert len(torch.unique(out_indices)) == len(out_indices)

    self.free_pages = self.free_pages[num_new_pages:]
    return out_indices

  def free(self, free_index: torch.Tensor):
    if free_index.numel() == 0:
      return

    if self.is_not_in_free_group:
      free_page_indices = torch.unique(free_index // self.page_size)
      # Page 0 is a reserved/padded slot and must never enter free pages.
      valid_mask = (free_page_indices >= 1) & (free_page_indices <= self.num_pages)
      if self.debug_mode and not bool(torch.all(valid_mask)):
        invalid = free_page_indices[~valid_mask]
        logger.warning("Dropping invalid free page ids: %s", invalid.detach().cpu().tolist())
      free_page_indices = free_page_indices[valid_mask]
      if free_page_indices.numel() == 0:
        return
      self.free_pages = torch.cat((free_page_indices, self.free_pages))
      # 去重：防止 radix tree split 节点时同一个 page 被多个节点持有
      self.free_pages = torch.unique(self.free_pages)
    else:
      self.free_group.append(free_index)

    if self.debug_mode:
      assert len(torch.unique(self.free_pages)) == len(self.free_pages)
      assert torch.all(self.free_pages >= 1) and torch.all(self.free_pages <= self.num_pages)
      assert len(self.free_pages) <= self.num_pages

  def begin_free_group(self):
    """Batch `free()` calls to amortize dedup/concat overhead."""
    self.is_not_in_free_group = False

  def end_free_group(self):
    """Flush any batched frees and return allocator to normal mode."""
    self.flush_free_group()
    self.is_not_in_free_group = True

  def flush_free_group(self):
    """Materialize `free_group` into `free_pages` once."""
    if not self.free_group:
      return

    free_index = torch.cat(self.free_group)
    self.free_group = []
    if free_index.numel() == 0:
      return

    free_page_indices = torch.unique(free_index // self.page_size)
    # Page 0 is a reserved/padded slot and must never enter free pages.
    valid_mask = (free_page_indices >= 1) & (free_page_indices <= self.num_pages)
    if self.debug_mode and not bool(torch.all(valid_mask)):
      invalid = free_page_indices[~valid_mask]
      logger.warning("Dropping invalid free page ids: %s", invalid.detach().cpu().tolist())
    free_page_indices = free_page_indices[valid_mask]
    if free_page_indices.numel() == 0:
      return

    # Merge into allocator free list with one dedup pass.
    self.free_pages = torch.unique(torch.cat((free_page_indices, self.free_pages)))

    if self.debug_mode:
      assert len(torch.unique(self.free_pages)) == len(self.free_pages)
      assert torch.all(self.free_pages >= 1) and torch.all(self.free_pages <= self.num_pages)
      assert len(self.free_pages) <= self.num_pages

  def clear(self):
    # The padded slot 0 is used for writing dummy outputs from padded tokens.
    self.free_pages = torch.arange(1, self.num_pages + 1, dtype=torch.int64, device=self.device)
    self.is_not_in_free_group = True
    self.free_group = []

  # ======= ITokenAllocator compatibility helpers =======
  def alloc_pages_extend(
    self,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
  ) -> Optional[torch.Tensor]:
    return self.alloc_extend(
      prefix_lens,
      prefix_lens_cpu,
      seq_lens,
      seq_lens_cpu,
      last_loc,
      extend_num_tokens,
    )

  def alloc_pages_decode(
    self,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
  ) -> Optional[torch.Tensor]:
    return self.alloc_decode(seq_lens, seq_lens_cpu, last_loc)

  def available_size(self) -> int:
    # Only count allocatable pages currently present in `free_pages`.
    # NOTE: `free_group` is not used by alloc paths in current implementation.
    free_pages = len(self.free_pages)
    return free_pages * self.page_size


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

  def req_to_token_pool(self):
    return self.req_to_token

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
