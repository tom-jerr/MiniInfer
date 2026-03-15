from __future__ import annotations
import math
import torch
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Tuple

try:
  import flashinfer
except ImportError:
  flashinfer = None

from .base_backend import AttentionBackend
from miniinfer.layers.attention import AttentionImpl
from miniinfer.scheduler.scheduler_batch import ForwardBatch

if TYPE_CHECKING:
  from miniinfer.kvcache.kv_cache_manager import KVCacheManager

logger = logging.getLogger(__name__)


@dataclass
class FlashInferMetadata:
  # Decoding wrappers
  # Shared across layers if heads/dim are constant, but typically wrappers are field of backend
  # Here we store batch-specific metadata

  # Common metadata
  paged_kv_indptr: torch.Tensor = None
  paged_kv_indices: torch.Tensor = None
  paged_kv_last_page_len: torch.Tensor = None

  # Extend specific
  qo_indptr: torch.Tensor = None  # query offsets

  # Decode specific
  # In decode, qo_indptr is just 0..batch
  pass


class FlashInferBackend(AttentionBackend):
  def __init__(self, kv_cache_mgr: KVCacheManager):
    super().__init__()
    self.kv_cache_mgr = kv_cache_mgr
    self.page_size = kv_cache_mgr.page_size
    self.device = kv_cache_mgr.device

    if flashinfer is None:
      raise ImportError("flashinfer is not installed")

    # Initialize FlashInfer wrappers
    # Allocate a workspace buffer (256MB is usually sufficient for typical batch sizes)
    self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=self.device)

    self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
      self.workspace_buffer, kv_layout="NHD"
    )
    self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
      self.workspace_buffer, kv_layout="NHD"
    )

    self.forward_metadata: FlashInferMetadata = None

  def type(self) -> str:
    return "flashinfer"

  def init_forward_metadata(self, forward_batch: ForwardBatch):
    metadata = FlashInferMetadata()

    batch_size = forward_batch.batch_size
    seq_lens = forward_batch.seq_lens

    # Determine max needed pages
    # forward_batch.max_seq_len is in tokens
    max_seq_len = forward_batch.max_seq_len
    # We need enough columns from page_table
    if max_seq_len is None:
      if forward_batch.seq_lens_cpu is not None:
        max_seq_len = int(forward_batch.seq_lens_cpu.max())
      else:
        max_seq_len = int(seq_lens.max().item())

    num_pages_max = (max_seq_len + self.page_size - 1) // self.page_size

    # Get raw page table [batch_size, max_num_pages] or [batch_size, max_tokens] depending on impl
    # Passing num_pages_max to get_page_table ensures we get enough columns if it returns blocks
    # But if get_page_table expects token count, we should pass max_seq_len.
    # Based on FA3 backend, it passes max_seq_len (tokens), so let's stick to that.
    # But we will slice it as needed.
    raw_page_table = self.kv_cache_mgr.get_page_table(forward_batch, max_seq_len)

    # Construct paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len
    paged_kv_indptr_list = [0]
    paged_kv_indices_list = []
    paged_kv_last_page_len_list = []

    # Helper to extract valid pages
    # This loop runs on CPU preferably, but converting tensors to CPU might be slow?
    # FlashInfer utilities usually take tensors.
    # Let's try to operation on tensors if possible, or assume simple logic

    # Optimization: Use torch operations to flatten
    # Calculate num_pages per request
    num_pages = (seq_lens + self.page_size - 1) // self.page_size  # [batch]
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
    torch.cumsum(num_pages, dim=0, out=paged_kv_indptr[1:])

    last_page_len = (seq_lens - 1) % self.page_size + 1
    metadata.paged_kv_last_page_len = last_page_len.to(torch.int32)
    metadata.paged_kv_indptr = paged_kv_indptr.to(torch.int32)

    # Flatten valid indices
    # We need to select valid pages from raw_page_table corresponding to each row
    # Can use masking
    max_pages_dim = raw_page_table.shape[1]
    col_indices = torch.arange(max_pages_dim, device=self.device).expand(batch_size, -1)
    mask = col_indices < num_pages.unsqueeze(1)
    metadata.paged_kv_indices = raw_page_table[mask].contiguous().to(torch.int32)

    if not forward_batch.forward_mode.is_decode():
      # Extend mode metadata
      if any(forward_batch.extend_prefix_lens_cpu):
        # Mixed extends ?
        # qo_indptr based on extend_seq_lens
        extend_seq_lens = forward_batch.extend_seq_lens
        metadata.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(extend_seq_lens, dim=0, out=metadata.qo_indptr[1:])
      else:
        # Standard extend where we process whole seq or suffix
        metadata.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(seq_lens, dim=0, out=metadata.qo_indptr[1:])  # if processing all?
        # Wait, in extend, q is [total_tokens, ...].
        # If we are extending, we usually process `extend_seq_lens` tokens.
        # FlashAttention3Backend uses `extend_seq_lens` if available.
        # Let's check FA3 logic again.
        if hasattr(forward_batch, "extend_seq_lens") and forward_batch.extend_seq_lens is not None:
          metadata.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
          torch.cumsum(forward_batch.extend_seq_lens, dim=0, out=metadata.qo_indptr[1:])
        else:
          # Fallback to seq_lens (e.g. initial prefill of full prompts)
          metadata.qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
          torch.cumsum(seq_lens, dim=0, out=metadata.qo_indptr[1:])

    self.forward_metadata = metadata

  def init_cuda_graph_metadata(
    self,
    batch_size: int,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seq_len_k: int,
  ) -> None:
    """Initialize metadata for CUDA graph capture."""
    self.forward_metadata = FlashInferMetadata()

    # Store references to static buffers provided by CudaGraphRunner
    self.forward_metadata.block_table = block_table  # [BS, MaxBlocks]
    self.forward_metadata.cache_seqlens = cache_seqlens  # [BS]

    # Allocate static buffers for FlashInfer CSR format
    # Max total pages = batch_size * max_blocks_per_req
    max_blocks_per_req = block_table.shape[1]
    max_total_pages = batch_size * max_blocks_per_req

    device = self.device

    # paged_kv_indptr: [batch_size + 1]
    self.forward_metadata.paged_kv_indptr = torch.zeros(
      batch_size + 1, dtype=torch.int32, device=device
    )

    # paged_kv_indices: [max_total_pages]
    self.forward_metadata.paged_kv_indices = torch.zeros(
      max_total_pages, dtype=torch.int32, device=device
    )

    # paged_kv_last_page_len: [batch_size]
    self.forward_metadata.paged_kv_last_page_len = torch.zeros(
      batch_size, dtype=torch.int32, device=device
    )

    # cu_seqlens_q for decode (0..BS)
    self.forward_metadata.cu_seqlens_q = torch.arange(
      0, batch_size + 1, dtype=torch.int32, device=device
    )

  def update_cuda_graph_metadata(
    self,
    cache_seqlens: torch.Tensor,
    max_seq_len_k: int,
  ) -> None:
    """Update metadata before graph replay."""
    # Note: cache_seqlens passed here is the same buffer as self.forward_metadata.cache_seqlens
    # but with updated content.

    meta = self.forward_metadata
    block_table = meta.block_table
    batch_size = block_table.shape[0]

    # Convert BlockTable + SeqLens -> CSR (IndPtr, Indices, LastPageLen)
    # This runs on GPU (PyTorch ops), outside the graph replay.

    # 1. Calculate num_pages per request
    # num_pages = (seq_len + page_size - 1) // page_size
    num_pages = (cache_seqlens + self.page_size - 1) // self.page_size

    # 2. Update paged_kv_indptr
    # meta.paged_kv_indptr[0] is already 0
    torch.cumsum(num_pages, dim=0, out=meta.paged_kv_indptr[1:])

    # 3. Update paged_kv_last_page_len
    # last_page_len = (seq_len - 1) % page_size + 1
    # Use torch.where to handle seq_len=0 case if necessary,
    # though decode usually has seq_len > 0.
    last_page_len = (cache_seqlens - 1) % self.page_size + 1
    meta.paged_kv_last_page_len.copy_(last_page_len)

    # 4. Update paged_kv_indices
    # We need to flatten the valid blocks from block_table
    # Flatten logic: select elements where col_index < num_pages
    # This produces a 1D tensor of valid indices.
    # We copy this 1D tensor into the pre-allocated meta.paged_kv_indices

    # Create mask [BS, MaxBlocks]
    max_blocks = block_table.shape[1]
    col_indices = torch.arange(max_blocks, device=self.device).expand(batch_size, -1)
    mask = col_indices < num_pages.unsqueeze(1)

    valid_indices = block_table[mask].contiguous()

    # Copy to static buffer.
    # Note: valid_indices length varies. We copy into the beginning of the buffer.
    # The graph will use indptr to know how many to read.
    # Ensure we don't overflow (shouldn't if max size is correct)

    # We must use slice assignment
    count = valid_indices.numel()
    meta.paged_kv_indices[:count].copy_(valid_indices)

  def forward_decode(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: AttentionImpl,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    **kwargs,
  ) -> torch.Tensor:
    # Save KV Cache
    if k is not None and save_kv_cache:
      assert v is not None
      cache_loc = (
        forward_batch.out_cache_loc
        # if not layer.is_cross_attention
        # else forward_batch.encoder_out_cache_loc
      )
      self.kv_cache_mgr.set_kv_buffer(
        layer.layer_id,
        cache_loc,
        k,
        v,
      )

    metadata = self.forward_metadata

    # Prepare Batch plan
    # We need to call begin_forward.
    # Note: In a multi-layer loop, calling this repeatedly with same args is overhead but
    # necessary if we don't assume layers are identical (though they usually are).
    # We can try to optimize by checking if plan is already valid?
    # wrappers check internally? 0.5.x wrappers are stateful.
    # We will call it every time to be safe.

    # During CUDA graph capture, we must not call begin_forward as it may involve CPU ops
    # We rely on the warmup run (executed just before capture) to have set up the wrapper state correctly
    if not torch.cuda.is_current_stream_capturing():
      self.decode_wrapper.begin_forward(
        metadata.paged_kv_indptr,
        metadata.paged_kv_indices,
        metadata.paged_kv_last_page_len,
        num_qo_heads=layer.tp_q_head_num,
        num_kv_heads=layer.tp_k_head_num,
        head_dim=layer.head_dim,
        page_size=self.page_size,
        data_type=q.dtype,  # assuming q dtype matches kv
      )

    # Get KV buffers
    key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)
    # Reshape to 4D/5D if needed or pass as is?
    # FlashInfer paged input expects: (num_pages, 2, num_heads, page_size, head_dim) OR tuple.
    # MiniInfer storage: (capacity, num_heads, head_dim)
    # We need to reshape them to mimic paged structure: (num_pages (capacity/page_size), page_size, num_heads, head_dim)
    # Then permute to (num_pages, num_heads, page_size, head_dim)

    # Wait, if buffer is flat (size + page_size), we must view it as pages.
    # capacity must be divisible by page_size? Not necessarily in MiniInfer's simpler allocator.
    # But if we access via paged_kv_indices which are PAGE indices,
    # then the underlying buffer MUST be viewable as pages.
    # If MiniInfer allocator produces arbitrary indices (token based), and we treat them as page indices? No.

    # If MiniInfer page_size=1, then "page" is 1 token.
    # key_cache view: (-1, 1, num_heads, head_dim).
    # Valid.

    # If MiniInfer page_size=16. Allocator gives block indices?
    # Checking FA3 backend: key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
    # This confirms the buffer is organized as blocks.

    key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim).permute(
      0, 2, 1, 3
    )
    value_cache = value_cache.view(
      -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
    ).permute(0, 2, 1, 3)
    # Shape now: (num_blocks, num_heads, page_size, head_dim)

    output = self.decode_wrapper.forward(
      q.view(-1, layer.tp_q_head_num, layer.head_dim),
      (key_cache, value_cache),
      sm_scale=layer.scaling,
    )

    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

  def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: AttentionImpl,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    **kwargs,
  ) -> torch.Tensor:
    if k is not None and save_kv_cache:
      cache_loc = forward_batch.out_cache_loc
      self.kv_cache_mgr.set_kv_buffer(
        layer.layer_id,
        cache_loc,
        k,
        v,
      )

    metadata = self.forward_metadata

    self.prefill_wrapper.begin_forward(
      metadata.qo_indptr,
      metadata.paged_kv_indptr,
      metadata.paged_kv_indices,
      metadata.paged_kv_last_page_len,
      num_qo_heads=layer.tp_q_head_num,
      num_kv_heads=layer.tp_k_head_num,
      head_dim_qk=layer.head_dim,
      page_size=self.page_size,
      q_data_type=q.dtype,
    )

    key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)
    # Reshape for paged access
    key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim).permute(
      0, 2, 1, 3
    )
    value_cache = value_cache.view(
      -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
    ).permute(0, 2, 1, 3)

    output = self.prefill_wrapper.forward(
      q.view(-1, layer.tp_q_head_num, layer.head_dim),
      (key_cache, value_cache),
      causal=True,
      sm_scale=layer.scaling,
    )

    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
