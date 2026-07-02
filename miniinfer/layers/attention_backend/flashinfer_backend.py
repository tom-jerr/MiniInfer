from __future__ import annotations
import math
import torch
from miniinfer.utils import get_logger
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

logger = get_logger(__name__)


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
    # 每个 cuda-graph capture batch size 一组静态 buffer（graph 冻结各自引用，
    # 不能跨 batch size 复用同一个 buffer，否则后捕获的 batch 会覆盖前者）。
    self._cg_buffers: dict = {}

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

    # raw_page_table: [batch, max_seq_len] 的 **token** 索引（每 token 一个 KV slot）。
    raw_page_table = self.kv_cache_mgr.get_page_table(forward_batch, max_seq_len)

    # FlashInfer 的 paged_kv_indices 要 **page** 索引（每页 1 个），不是 token 索引。
    # 取每页起始 token（每隔 page_size 列）再 // page_size 得到 page 索引。
    page_table = raw_page_table[:, :: self.page_size] // self.page_size  # [batch, max_num_pages]

    # num_pages per request
    num_pages = (seq_lens + self.page_size - 1) // self.page_size  # [batch]
    paged_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
    torch.cumsum(num_pages, dim=0, out=paged_kv_indptr[1:])

    last_page_len = (seq_lens - 1) % self.page_size + 1
    metadata.paged_kv_last_page_len = last_page_len.to(torch.int32)
    metadata.paged_kv_indptr = paged_kv_indptr.to(torch.int32)

    # Flatten valid page indices (col < num_pages per row) into 1D.
    max_pages_dim = page_table.shape[1]
    col_indices = torch.arange(max_pages_dim, device=self.device).expand(batch_size, -1)
    mask = col_indices < num_pages.unsqueeze(1)
    metadata.paged_kv_indices = page_table[mask].contiguous().to(torch.int32)

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
    """Initialize metadata for CUDA graph capture.

    静态 buffer 存在 ``self._cg_*`` 上（而非 ``self.forward_metadata``），因为
    prefill 步的 ``init_forward_metadata`` 会替换 ``forward_metadata``，若存在
    ``forward_metadata`` 上会被清掉。capture 时把 ``forward_metadata.paged_kv_*``
    指向这些静态 buffer，graph 冻结引用；replay 前 ``update_cuda_graph_metadata``
    原地更新这些静态 buffer，graph 即读到新值。
    """
    max_blocks_per_req = block_table.shape[1]
    max_total_pages = batch_size * max_blocks_per_req
    device = self.device

    # 静态 buffer（CudaGraphRunner.replay 会原地把实际 block_table / cache_seqlens
    # copy 进 block_table / cache_seqlens 这两个传入的 buffer）。按 batch_size 存，
    # 每个 capture size 的 graph 冻结各自一组。
    indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    indices = torch.zeros(max_total_pages, dtype=torch.int32, device=device)
    last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    self._cg_buffers[batch_size] = {
      "block_table": block_table,  # [BS, MaxBlocks] page indices
      "cache_seqlens": cache_seqlens,  # [BS]
      "paged_kv_indptr": indptr,
      "paged_kv_indices": indices,
      "paged_kv_last_page_len": last_page_len,
    }

    # forward_metadata 指向本 batch_size 的静态 buffer，供 capture 冻结引用。
    self.forward_metadata = FlashInferMetadata()
    self.forward_metadata.block_table = block_table
    self.forward_metadata.cache_seqlens = cache_seqlens
    self.forward_metadata.paged_kv_indptr = indptr
    self.forward_metadata.paged_kv_indices = indices
    self.forward_metadata.paged_kv_last_page_len = last_page_len
    # cu_seqlens_q for decode (0..BS)
    self.forward_metadata.cu_seqlens_q = torch.arange(
      0, batch_size + 1, dtype=torch.int32, device=device
    )

  def update_cuda_graph_metadata(
    self,
    cache_seqlens: torch.Tensor,
    max_seq_len_k: int,
  ) -> None:
    """Update metadata before graph replay.

    用对应 batch_size 的静态 buffer（``self._cg_buffers[batch_size]``），从
    block_table + cache_seqlens 重建 CSR，原地写回 paged_kv_*。graph capture 时
    冻结的就是该 batch_size 的 buffer，故能读到新值。
    """
    batch_size = cache_seqlens.shape[0]
    bufs = self._cg_buffers[batch_size]
    block_table = bufs["block_table"]

    # Convert BlockTable + SeqLens -> CSR (IndPtr, Indices, LastPageLen)
    # This runs on GPU (PyTorch ops), outside the graph replay.

    # 1. Calculate num_pages per request
    # num_pages = (seq_len + page_size - 1) // page_size
    num_pages = (cache_seqlens + self.page_size - 1) // self.page_size

    # 2. Update paged_kv_indptr (in-place into static buffer)
    torch.cumsum(num_pages, dim=0, out=bufs["paged_kv_indptr"][1:])

    # 3. Update paged_kv_last_page_len
    last_page_len = (cache_seqlens - 1) % self.page_size + 1
    bufs["paged_kv_last_page_len"].copy_(last_page_len)

    # 4. Update paged_kv_indices: flatten valid page indices (col < num_pages).
    max_blocks = block_table.shape[1]
    col_indices = torch.arange(max_blocks, device=self.device).expand(batch_size, -1)
    mask = col_indices < num_pages.unsqueeze(1)
    valid_indices = block_table[mask].contiguous()

    count = valid_indices.numel()
    idx_buf = bufs["paged_kv_indices"]
    idx_buf[:count].copy_(valid_indices)
    # 清掉尾部残留（indptr 已限定范围，非必须，但更稳妥）。
    if count < idx_buf.numel():
      idx_buf[count:].zero_()

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
    # FlashInfer kv_layout="NHD" 期望 [num_pages, page_size, num_kv_heads, head_dim]。
    # MiniInfer KV buffer 是 [capacity, num_heads, head_dim]，view 成
    # [num_pages, page_size, num_heads, head_dim] 即 NHD，**不要** permute 成 HND
    # （否则与声明的 NHD 不符，kernel 读到错位数据 → segfault / "group_size: 0"）。
    key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
    value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)

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
    # NHD layout: [num_pages, page_size, num_heads, head_dim]（与 decode 一致，不 permute）。
    key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
    value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)

    output = self.prefill_wrapper.forward(
      q.view(-1, layer.tp_q_head_num, layer.head_dim),
      (key_cache, value_cache),
      causal=True,
      sm_scale=layer.scaling,
    )

    return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)
