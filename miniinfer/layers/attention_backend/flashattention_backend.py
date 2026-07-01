from __future__ import annotations
from .base_backend import AttentionBackend
import torch
from dataclasses import dataclass
from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
from typing import Optional, TYPE_CHECKING
from miniinfer.utils import get_logger

if TYPE_CHECKING:
  from miniinfer.layers.attention import AttentionImpl
  from miniinfer.scheduler.scheduler_batch import ForwardBatch
  from miniinfer.kvcache.kv_cache_manager import KVCacheManager

logger = get_logger(__name__)


@dataclass
class FlashAttention3Metadata:
  """Metadata to be init once in the model forward pass,
  each layer's forward pass can reuse the metadata.

  For each init metadata function, we will try set up them in below order
  """

  # Sequence lengths for the forward batch
  cache_seqlens_int32: torch.Tensor = None
  # Maximum sequence length for query
  max_seq_len_q: int = 1
  # Maximum sequence length for key
  max_seq_len_k: int = 0
  # Cumulative sequence lengths for query
  cu_seqlens_q: torch.Tensor = None
  # Cumulative sequence lengths for key
  cu_seqlens_k: torch.Tensor = None
  # Page table, the index of KV Cache Tables/Blocks
  page_table: torch.Tensor = None


class FlashAttention3Backend(AttentionBackend):
  def __init__(self, kv_cache_mgr: KVCacheManager):
    super().__init__()
    self.forward_metadata: FlashAttention3Metadata = None
    self.kv_cache_mgr: KVCacheManager = kv_cache_mgr
    self.page_size: int = kv_cache_mgr.page_size

  def type(self) -> str:
    return "flash_attention3"

  def init_forward_metadata(self, forward_batch: "ForwardBatch"):
    metadata = FlashAttention3Metadata()
    seqlens_in_batch = forward_batch.seq_lens
    batch_size = forward_batch.batch_size
    device = seqlens_in_batch.device

    # Use precomputed int32 version if available
    seqlens_int32 = (
      forward_batch.seq_lens_int32
      if forward_batch.seq_lens_int32 is not None
      else seqlens_in_batch.to(torch.int32)
    )

    if forward_batch.forward_mode.is_decode():
      cache_seqlens = seqlens_int32
      if getattr(forward_batch, "debug_use_past_len_cache", False):
        cache_seqlens = torch.clamp(cache_seqlens - 1, min=0)
      metadata.cache_seqlens_int32 = cache_seqlens
      # Use precomputed max_seq_len from CPU to avoid GPU sync
      metadata.max_seq_len_k = forward_batch.max_seq_len
      if metadata.max_seq_len_k is None:
        # Fallback: compute from CPU tensor if available, else GPU tensor
        if forward_batch.seq_lens_cpu is not None:
          metadata.max_seq_len_k = int(forward_batch.seq_lens_cpu.max())
        else:
          raise RuntimeError(
            "FlashAttention3Backend.init_forward_metadata: missing CPU-side seq_lens metadata "
            "(forward_batch.max_seq_len / forward_batch.seq_lens_cpu). "
            "Refusing to fall back to CUDA `.item()` which would sync and break overlap."
          )
      metadata.cu_seqlen_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
      metadata.cu_seqlen_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
      )
      metadata.page_table = self.kv_cache_mgr.get_page_table(forward_batch, metadata.max_seq_len_k)
    else:
      metadata.cache_seqlens_int32 = seqlens_int32
      # Use precomputed max_seq_len from CPU to avoid GPU sync
      metadata.max_seq_len_q = forward_batch.max_seq_len
      if metadata.max_seq_len_q is None:
        # Fallback: compute from CPU tensor if available, else GPU tensor
        if forward_batch.seq_lens_cpu is not None:
          metadata.max_seq_len_q = int(forward_batch.seq_lens_cpu.max())
        else:
          raise RuntimeError(
            "FlashAttention3Backend.init_forward_metadata: missing CPU-side seq_lens metadata "
            "(forward_batch.max_seq_len / forward_batch.seq_lens_cpu). "
            "Refusing to fall back to CUDA `.item()` which would sync and break overlap."
          )

      metadata.max_seq_len_k = metadata.max_seq_len_q
      metadata.cu_seqlens_q = torch.nn.functional.pad(
        torch.cumsum(seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
      )
      metadata.cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
      )
      metadata.page_table = self.kv_cache_mgr.get_page_table(forward_batch, metadata.max_seq_len_k)
      if any(forward_batch.extend_prefix_lens_cpu):
        extend_seq_lens = forward_batch.extend_seq_lens
        metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
        metadata.cu_seqlens_q = torch.nn.functional.pad(
          torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
      else:
        metadata.max_seq_len_q = metadata.max_seq_len_k
        metadata.cu_seqlens_q = metadata.cu_seqlens_k

    self.forward_metadata = metadata

  def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: AttentionImpl,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    **kwargs,
  ):
    if k is not None:
      assert v is not None
      if save_kv_cache:
        cache_loc = forward_batch.out_cache_loc

        self.kv_cache_mgr.set_kv_buffer(layer, cache_loc, k, v, layer.k_scale, layer.v_scale)

    # Use precomputed metadata across all layers
    metadata = self.forward_metadata
    causal = True
    page_table = metadata.page_table
    cu_seqlens_q = metadata.cu_seqlens_q
    cache_seqlens = metadata.cache_seqlens_int32
    max_seqlen_q = metadata.max_seq_len_q
    cu_seqlens_k = metadata.cu_seqlens_k

    # Do multi-head attention
    key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
    value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)

    o = flash_attn_with_kvcache(
      q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
      k_cache=key_cache,
      v_cache=value_cache,
      page_table=page_table,
      cache_seqlens=cache_seqlens,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_k_new=cu_seqlens_k,
      max_seqlen_q=max_seqlen_q,
      softmax_scale=layer.scaling,
      causal=causal,
      # softcap=layer.logit_cap,
      **kwargs,
    )

    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

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
    assert self.fa_impl_ver in [3], "Only FA3 support decoding"
    if k is not None:
      assert v is not None
      if save_kv_cache:
        cache_loc = (
          forward_batch.out_cache_loc
          if not layer.is_cross_attention
          else forward_batch.encoder_out_cache_loc
        )

        self.kv_cache_mgr.set_kv_buffer(
          layer.layer_id,
          cache_loc,
          k,
          v,
        )

    # Use precomputed metadata across all layers
    metadata = self.forward_metadata
    causal = True

    # Do multi-head attention

    key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)
    key_cache = key_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
    value_cache = value_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)

    page_table = metadata.page_table
    cache_seqlens = metadata.cache_seqlens_int32
    # TODO: maybe decode only needs cu_seqlens_q
    cu_seqlens_k = metadata.cu_seqlens_k
    max_seqlen_q = metadata.max_seq_len_q
    q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

    # Default: single-token self-attention
    o = flash_attn_with_kvcache(
      q=q_reshaped,
      k_cache=key_cache,
      v_cache=value_cache,
      page_table=page_table,
      cache_seqlens=cache_seqlens,
      cu_seqlens_q=metadata.cu_seqlens_q,
      max_seqlen_q=max_seqlen_q,
      softmax_scale=layer.scaling,
      causal=causal,
      # softcap=layer.logit_cap,
      **kwargs,
    )

    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


@dataclass
class FlashAttention2Metadata:
  """Metadata to be init once in the model forward pass,
  each layer's forward pass can reuse the metadata.

  For each init metadata function, we will try set up them in below order
  """

  # Sequence lengths for the forward batch
  cache_seqlens_int32: torch.Tensor = None
  # Maximum sequence length for query
  max_seq_len_q: int = 1
  # Maximum sequence length for key
  max_seq_len_k: int = 0
  # Cumulative sequence lengths for query
  cu_seqlens_q: torch.Tensor = None
  # Cumulative sequence lengths for key
  cu_seqlens_k: torch.Tensor = None
  # Page table, the index of KV Cache Tables/Blocks
  block_table: torch.Tensor = None


class FlashAttention2Backend(AttentionBackend):
  def __init__(self, kv_cache_mgr: KVCacheManager):
    super().__init__()
    self.forward_metadata: FlashAttention2Metadata = None
    self.kv_cache_mgr: KVCacheManager = kv_cache_mgr
    self.page_size: int = kv_cache_mgr.page_size

  def type(self) -> str:
    return "flash_attention2"

  def init_forward_metadata(self, forward_batch: "ForwardBatch"):
    metadata = FlashAttention2Metadata()
    seqlens_in_batch = forward_batch.seq_lens
    batch_size = forward_batch.batch_size
    device = seqlens_in_batch.device

    # Use precomputed int32 version if available
    seqlens_int32 = (
      forward_batch.seq_lens_int32
      if forward_batch.seq_lens_int32 is not None
      else seqlens_in_batch.to(torch.int32)
    )

    def _build_block_table(max_seq_len_k: int) -> torch.Tensor:
      # RequestPool stores per-token KV indices (kv_loc). FlashAttention expects per-block indices.
      # Each sequence is laid out in pages of `page_size`, so the block id is kv_loc // page_size.
      kv_locs = self.kv_cache_mgr.get_page_table(forward_batch, max_seq_len_k)
      # Debug: check kv_locs
      if getattr(forward_batch, "debug_decode", False):
        logger.debug(f"[_build_block_table] max_seq_len_k: {max_seq_len_k}")
        logger.debug(f"[_build_block_table] kv_locs shape: {kv_locs.shape}")
        logger.debug(f"[_build_block_table] kv_locs[:, :5]: {kv_locs[:, :5].tolist()}")
      # Take the first token of each page to form the block table.
      # Flash Attention requires block_table to have contiguous last dimension.
      # Strided slicing (::page_size) creates non-contiguous memory, so we must
      # call contiguous() right after the slice before any further operations.
      block_indices = kv_locs[:, :: self.page_size].contiguous()
      block_table = (block_indices // self.page_size).to(torch.int32).contiguous()
      if getattr(forward_batch, "debug_decode", False):
        logger.debug(f"[_build_block_table] block_table: {block_table.tolist()}")
      return block_table

    if forward_batch.forward_mode.is_decode():
      cache_seqlens = seqlens_int32
      if getattr(forward_batch, "debug_use_past_len_cache", False):
        cache_seqlens = torch.clamp(cache_seqlens - 1, min=0)
      metadata.cache_seqlens_int32 = cache_seqlens
      # Use precomputed max_seq_len from CPU to avoid GPU sync
      metadata.max_seq_len_k = forward_batch.max_seq_len
      if metadata.max_seq_len_k is None:
        if forward_batch.seq_lens_cpu is not None:
          metadata.max_seq_len_k = int(forward_batch.seq_lens_cpu.max())
        else:
          raise RuntimeError(
            "FlashAttention2Backend.init_forward_metadata: missing CPU-side seq_lens metadata "
            "(forward_batch.max_seq_len / forward_batch.seq_lens_cpu). "
            "Refusing to fall back to CUDA `.item()` which would sync and break overlap."
          )
      # Note: For decode, each sequence has seqlen_q=1
      metadata.max_seq_len_q = 1
      metadata.cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
      metadata.cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
      )
      metadata.block_table = _build_block_table(metadata.max_seq_len_k)
    else:
      metadata.cache_seqlens_int32 = seqlens_int32
      # Use precomputed max_seq_len from CPU to avoid GPU sync
      metadata.max_seq_len_q = forward_batch.max_seq_len
      if metadata.max_seq_len_q is None:
        if forward_batch.seq_lens_cpu is not None:
          metadata.max_seq_len_q = int(forward_batch.seq_lens_cpu.max())
        else:
          raise RuntimeError(
            "FlashAttention2Backend.init_forward_metadata: missing CPU-side seq_lens metadata "
            "(forward_batch.max_seq_len / forward_batch.seq_lens_cpu). "
            "Refusing to fall back to CUDA `.item()` which would sync and break overlap."
          )
      metadata.max_seq_len_k = metadata.max_seq_len_q
      metadata.cu_seqlens_q = torch.nn.functional.pad(
        torch.cumsum(seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
      )
      metadata.cu_seqlens_k = torch.nn.functional.pad(
        torch.cumsum(seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
      )
      metadata.block_table = _build_block_table(metadata.max_seq_len_k)
      if forward_batch.extend_prefix_lens_cpu and any(forward_batch.extend_prefix_lens_cpu):
        extend_seq_lens = forward_batch.extend_seq_lens
        if not isinstance(extend_seq_lens, torch.Tensor):
          # 优化：使用 torch.as_tensor 避免 CUDA 同步
          extend_seq_lens_cpu = torch.as_tensor(extend_seq_lens, dtype=torch.int32)
          extend_seq_lens = extend_seq_lens_cpu.to(device, non_blocking=True)
        else:
          extend_seq_lens = extend_seq_lens.to(device=device, dtype=torch.int32, non_blocking=True)
        metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
        metadata.cu_seqlens_q = torch.nn.functional.pad(
          torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
        )
      else:
        metadata.max_seq_len_q = metadata.max_seq_len_k
        metadata.cu_seqlens_q = metadata.cu_seqlens_k

    self.forward_metadata = metadata

  def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: AttentionImpl,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    **kwargs,
  ):
    if k is not None:
      assert v is not None
      if save_kv_cache:
        cache_loc = forward_batch.out_cache_loc

        self.kv_cache_mgr.set_kv_buffer(layer.layer_id, cache_loc, k, v)

    # Use precomputed metadata across all layers
    metadata = self.forward_metadata
    causal = True
    cu_seqlens_q = metadata.cu_seqlens_q
    max_seqlen_q = metadata.max_seq_len_q
    cu_seqlens_k = metadata.cu_seqlens_k

    use_prefix_cache = bool(
      forward_batch.extend_prefix_lens_cpu and any(forward_batch.extend_prefix_lens_cpu)
    )

    q_reshaped = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
    if use_prefix_cache:
      # When prefix cache hits, `q/k/v` only contain the *new* (extend) tokens.
      # But `cu_seqlens_k` / `max_seq_len_k` are built from the *full* sequence lengths.
      # We must therefore gather the full K/V from the paged KV cache, otherwise FA will
      # read out-of-bounds (often surfacing later as a CUBLAS error).
      key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)
      max_seq_len_k = int(metadata.max_seq_len_k)
      kv_locs = self.kv_cache_mgr.get_page_table(forward_batch, max_seq_len_k)
      seq_lens = forward_batch.seq_lens.to(device=kv_locs.device)
      pos = torch.arange(max_seq_len_k, device=kv_locs.device).view(1, -1)
      mask = pos < seq_lens.view(-1, 1)
      packed_kv_locs = kv_locs[mask].to(torch.int64)
      k_reshaped = key_cache[packed_kv_locs].contiguous()
      v_reshaped = value_cache[packed_kv_locs].contiguous()
    else:
      # No prefix reuse: K/V are exactly the tokens in this forward, so use them directly.
      k_reshaped = k.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim)
      v_reshaped = v.contiguous().view(-1, layer.tp_v_head_num, layer.v_head_dim)

    o = flash_attn_varlen_func(
      q=q_reshaped,
      k=k_reshaped,
      v=v_reshaped,
      cu_seqlens_q=cu_seqlens_q,
      cu_seqlens_k=cu_seqlens_k,
      max_seqlen_q=max_seqlen_q,
      max_seqlen_k=metadata.max_seq_len_k,
      softmax_scale=layer.scaling,
      causal=causal,
      **kwargs,
    )

    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

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
    # Debug: print shapes for first layer
    debug_decode = getattr(forward_batch, "debug_decode", False) and layer.layer_id == 0
    if debug_decode:
      logger.debug("[FA2 forward_decode layer 0]")
      logger.debug(f"q shape: {q.shape}")
      logger.debug(f"k shape: {k.shape}")
      logger.debug(f"batch_size: {forward_batch.batch_size}")
      logger.debug(f"out_cache_loc: {forward_batch.out_cache_loc.tolist()}")

    if k is not None:
      assert v is not None
      if save_kv_cache:
        cache_loc = forward_batch.out_cache_loc

        self.kv_cache_mgr.set_kv_buffer(
          layer.layer_id,
          cache_loc,
          k,
          v,
        )

    # Use precomputed metadata across all layers
    metadata = self.forward_metadata
    causal = True

    # Do multi-head attention
    key_cache, value_cache = self.kv_cache_mgr.get_kv_buffer(layer.layer_id)

    if debug_decode:
      logger.debug(f"key_cache original shape: {key_cache.shape}")
      logger.debug(f"metadata.block_table: {metadata.block_table.tolist()}")
      logger.debug(f"metadata.cache_seqlens: {metadata.cache_seqlens_int32.tolist()}")

    key_cache = key_cache.view(
      -1, self.page_size, layer.tp_k_head_num, layer.head_dim
    )  # [N_pages, page_size, nheads, headdim]
    value_cache = value_cache.view(
      -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
    )  # [N_pages, page_size, nheads, headdim]

    if debug_decode:
      logger.debug(f"key_cache view shape: {key_cache.shape}")

    block_table = metadata.block_table
    cache_seqlens = metadata.cache_seqlens_int32
    batch_size = forward_batch.batch_size

    # Reshape q: flash_attn_with_kvcache expects [batch, seqlen_q, nheads, headdim]
    # For decode, seqlen_q=1
    q_reshaped = q.contiguous().view(batch_size, 1, layer.tp_q_head_num, layer.head_dim)

    if debug_decode:
      logger.debug(f"q_reshaped shape: {q_reshaped.shape}")

    # FA2 decode: use flash_attn_with_kvcache
    # Include cu_seqlens_q and max_seqlen_q for proper batch handling
    o = flash_attn_with_kvcache(
      q=q_reshaped,
      k_cache=key_cache,
      v_cache=value_cache,
      cache_seqlens=cache_seqlens,
      block_table=block_table,
      softmax_scale=layer.scaling,
      causal=causal,
      **kwargs,
    )

    if debug_decode:
      logger.debug(f"output shape: {o.shape}")

    if isinstance(o, tuple):
      o = o[0]
    # Output shape: [batch_size, seqlen_q=1, nheads, headdim]
    return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

  # ==================== CUDA Graph Support ====================

  def init_cuda_graph_metadata(
    self,
    batch_size: int,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seq_len_k: int,
  ) -> None:
    """
    Initialize metadata for CUDA graph capture.

    Unlike init_forward_metadata(), this uses pre-allocated static tensors
    that will be reused during graph replay. The tensors are NOT copied -
    they are used directly as references so updates propagate.

    Args:
        batch_size: Number of sequences in the batch
        block_table: Pre-allocated block table [batch_size, max_blocks]
        cache_seqlens: Pre-allocated sequence lengths [batch_size]
        max_seq_len_k: Maximum sequence length for keys
    """
    device = cache_seqlens.device

    metadata = FlashAttention2Metadata()
    # Use the provided tensors directly (not copies)
    metadata.cache_seqlens_int32 = cache_seqlens
    metadata.max_seq_len_k = max_seq_len_k
    metadata.max_seq_len_q = 1  # Decode always has seqlen_q=1
    # cu_seqlens_q for decode: simple 0, 1, 2, ..., batch_size
    metadata.cu_seqlens_q = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
    # cu_seqlens_k from cumsum of cache_seqlens
    metadata.cu_seqlens_k = torch.nn.functional.pad(
      torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    metadata.block_table = block_table

    self.forward_metadata = metadata

  def update_cuda_graph_metadata(
    self,
    cache_seqlens: torch.Tensor,
    max_seq_len_k: int,
  ) -> None:
    """
    Update metadata values before CUDA graph replay.

    This is called after copying actual sequence data into static buffers
    but before graph replay. It updates derived values that depend on the
    actual sequence lengths.

    Note: block_table and cache_seqlens_int32 are already updated via
    direct buffer copies. This method updates computed values.

    Args:
        cache_seqlens: Updated sequence lengths
        max_seq_len_k: Updated maximum sequence length
    """
    if self.forward_metadata is None:
      raise RuntimeError("forward_metadata not initialized. Call init_cuda_graph_metadata first.")

    self.forward_metadata.max_seq_len_k = max_seq_len_k
    # Update cu_seqlens_k with new cache_seqlens
    self.forward_metadata.cu_seqlens_k = torch.nn.functional.pad(
      torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
