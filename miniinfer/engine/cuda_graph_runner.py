"""
CUDA Graph Runner for Decode Phase

Captures and replays CUDA graphs for decode (single-token generation) to eliminate
kernel launch overhead. Follows SGLang-style implementation with:
- Pre-captured graphs for power-of-2 batch sizes
- Static placeholder buffers for graph capture/replay
- Padding to nearest power-of-2 for non-power-of-2 batch sizes

Usage:
    runner = CudaGraphRunner(model, attn_backend, kv_cache_mgr, model_config, max_bs=256)
    runner.warmup()  # Capture all graphs
    output = runner.replay(forward_batch)  # Run with CUDA graph
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from miniinfer.utils.profiler_utils import profile_methods

if TYPE_CHECKING:
    from miniinfer.models.fused_qwen2 import Qwen2ForCausalLM
    from miniinfer.layers.attention_backend.flashattention_backend import (
        FlashAttention2Backend,
    )
    from miniinfer.kvcache.kv_cache_manager import KVCacheManager
    from miniinfer.scheduler.scheduler_batch import ForwardBatch
    from miniinfer.models.base import BaseModelOutput


logger = logging.getLogger(__name__)


@dataclass
class CapturedGraph:
    """Stores a captured CUDA graph and its associated buffers."""

    graph: torch.cuda.CUDAGraph
    # Static input buffers (used during capture, reused during replay)
    input_ids: torch.Tensor  # [batch_size]
    positions: torch.Tensor  # [batch_size]
    block_table: torch.Tensor  # [batch_size, max_blocks_per_seq]
    cache_seqlens: torch.Tensor  # [batch_size]
    out_cache_loc: torch.Tensor  # [batch_size]
    req_pool_indices: torch.Tensor  # [batch_size]
    # Static output buffer
    output_logits: torch.Tensor  # [batch_size, vocab_size]
    # seq_lens as int64 (pre-converted to avoid conversion during replay)
    seq_lens_int64: torch.Tensor = None  # [batch_size]
    # ForwardBatch used during graph capture (reused during replay)
    graph_batch: Any = None


@profile_methods("CudaGraphRunner")
class CudaGraphRunner:
    """
    Manages CUDA graph capture and replay for decode phase.

    Captures graphs at power-of-2 batch sizes from 1 to max_batch_size.
    During inference, pads actual batch to nearest power-of-2 and replays
    the corresponding pre-captured graph.
    """

    def __init__(
        self,
        model: "Qwen2ForCausalLM",
        attn_backend: "FlashAttention2Backend",
        kv_cache_mgr: "KVCacheManager",
        max_batch_size: int = 256,
        max_context_len: int = 4096,
        page_size: int = 256,
        vocab_size: int = 151936,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """
        Initialize CUDA Graph Runner.

        Args:
            model: The LLM model to capture
            attn_backend: FlashAttention backend for attention computation
            kv_cache_mgr: KV cache manager
            max_batch_size: Maximum batch size to capture (should be max_num_seqs)
            max_context_len: Maximum context length for block table sizing
            page_size: KV cache page size
            vocab_size: Vocabulary size for output buffer
            dtype: Data type for tensors
            device: Device to run on
        """
        self.model = model
        self.attn_backend = attn_backend
        self.kv_cache_mgr = kv_cache_mgr
        self.max_batch_size = max_batch_size
        self.max_context_len = max_context_len
        self.page_size = page_size
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.device = device

        # Calculate max blocks per sequence
        self.max_blocks_per_seq = math.ceil(max_context_len / page_size)

        # Storage for captured graphs: {batch_size: CapturedGraph}
        self.graphs: Dict[int, CapturedGraph] = {}

        # Generate list of batch sizes to capture (powers of 2)
        self.capture_batch_sizes = self._get_capture_batch_sizes()

        logger.info(
            f"CudaGraphRunner initialized: max_bs={max_batch_size}, "
            f"capture_sizes={self.capture_batch_sizes}, "
            f"max_blocks_per_seq={self.max_blocks_per_seq}"
        )

    def _get_capture_batch_sizes(self) -> List[int]:
        """Get list of power-of-2 batch sizes to capture."""
        sizes = []
        bs = 1
        while bs <= self.max_batch_size:
            sizes.append(bs)
            bs *= 2
        # Ensure max_batch_size is included if it's not a power of 2
        if sizes[-1] < self.max_batch_size:
            sizes.append(self.max_batch_size)
        return sizes

    def _get_padded_batch_size(self, batch_size: int) -> int:
        """
        Get the nearest captured batch size (>= actual batch size).

        For non-power-of-2 batch sizes, returns the next power of 2.
        """
        if batch_size <= 0:
            return 1
        if batch_size > self.max_batch_size:
            return self.max_batch_size

        # Find smallest power of 2 >= batch_size
        padded = 1
        while padded < batch_size:
            padded *= 2
        return min(padded, self.max_batch_size)

    def _create_static_buffers(self, batch_size: int) -> CapturedGraph:
        """Create static placeholder buffers for a given batch size."""
        device = self.device

        return CapturedGraph(
            graph=torch.cuda.CUDAGraph(),
            input_ids=torch.zeros(batch_size, dtype=torch.int64, device=device),
            positions=torch.zeros(batch_size, dtype=torch.int64, device=device),
            block_table=torch.zeros(
                batch_size, self.max_blocks_per_seq, dtype=torch.int32, device=device
            ),
            cache_seqlens=torch.zeros(batch_size, dtype=torch.int32, device=device),
            out_cache_loc=torch.zeros(batch_size, dtype=torch.int64, device=device),
            req_pool_indices=torch.zeros(batch_size, dtype=torch.int64, device=device),
            output_logits=torch.zeros(
                batch_size, self.vocab_size, dtype=self.dtype, device=device
            ),
            seq_lens_int64=torch.zeros(batch_size, dtype=torch.int64, device=device),
        )

    def _capture_graph(self, batch_size: int) -> None:
        """
        Capture a CUDA graph for the given batch size.

        This creates a dummy forward pass and captures it into a CUDA graph.
        The static buffers are filled with placeholder values during capture.
        """
        logger.info(f"Capturing CUDA graph for batch_size={batch_size}...")

        # Create static buffers for this batch size
        captured = self._create_static_buffers(batch_size)

        # Fill with dummy but valid values for capture
        # input_ids: any valid token id (0 is usually safe)
        captured.input_ids.fill_(0)
        # positions: sequential for decode (each sequence at position seq_len-1)
        captured.positions.fill_(1)  # minimum valid position
        # block_table: point to block 0 (must be valid block index)
        captured.block_table.fill_(0)
        # cache_seqlens: minimum valid sequence length
        captured.cache_seqlens.fill_(2)  # at least 2 tokens
        # out_cache_loc: valid cache locations
        for i in range(batch_size):
            captured.out_cache_loc[i] = i
        # req_pool_indices: valid indices
        for i in range(batch_size):
            captured.req_pool_indices[i] = i

        # Pre-compute max_seq_len_k BEFORE graph capture (can't call .item() during capture)
        max_seq_len_k = 2  # Fixed dummy value for capture

        # Initialize seq_lens_int64 from cache_seqlens (using pre-allocated buffer)
        captured.seq_lens_int64.copy_(captured.cache_seqlens.to(torch.int64))

        # Prepare metadata for attention backend (static version)
        self.attn_backend.init_cuda_graph_metadata(
            batch_size=batch_size,
            block_table=captured.block_table,
            cache_seqlens=captured.cache_seqlens,
            max_seq_len_k=max_seq_len_k,
        )

        # Warm up run (required before graph capture)
        torch.cuda.synchronize()

        # Import here to avoid circular imports
        from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode

        # Run a warm-up forward pass
        with torch.no_grad():
            dummy_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=batch_size,
                input_ids=captured.input_ids,
                positions=captured.positions,
                req_pool_indices=captured.req_pool_indices,
                out_cache_loc=captured.out_cache_loc,
                seq_lens=captured.seq_lens_int64,
                all_seqs=None,  # Not needed for forward pass
                attn_backend=self.attn_backend,
            )
            # Warm-up run
            _ = self.model.forward(
                captured.input_ids,
                captured.positions,
                dummy_batch,
                return_hidden_states=False,
            )

        torch.cuda.synchronize()

        # Create the ForwardBatch BEFORE graph capture to avoid any Python operations during capture
        # The batch object itself is just a container and won't cause graph issues
        graph_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=captured.input_ids,
            positions=captured.positions,
            req_pool_indices=captured.req_pool_indices,
            out_cache_loc=captured.out_cache_loc,
            seq_lens=captured.seq_lens_int64,
            all_seqs=None,
            attn_backend=self.attn_backend,
        )

        # Store the graph_batch for replay (will update its tensors via the captured buffers)
        captured.graph_batch = graph_batch

        # Capture the graph - ONLY pure tensor operations inside
        with torch.cuda.graph(captured.graph):
            output = self.model.forward(
                captured.input_ids,
                captured.positions,
                graph_batch,
                return_hidden_states=False,
            )
            # Store output in static buffer
            captured.output_logits.copy_(output.logits)

        self.graphs[batch_size] = captured
        logger.info(f"CUDA graph captured for batch_size={batch_size}")

    def warmup(self) -> None:
        """
        Capture CUDA graphs for all power-of-2 batch sizes.

        Should be called once during engine initialization after model
        and KV cache are fully initialized.
        """
        logger.info(
            f"Starting CUDA graph warmup for batch sizes: {self.capture_batch_sizes}"
        )

        # Ensure we're in eval mode
        self.model.eval()

        # Capture graphs from largest to smallest (better memory allocation)
        for batch_size in reversed(self.capture_batch_sizes):
            self._capture_graph(batch_size)

        torch.cuda.synchronize()
        logger.info(f"CUDA graph warmup complete. Captured {len(self.graphs)} graphs.")

    def replay(self, forward_batch: "ForwardBatch") -> "BaseModelOutput":
        """
        Execute decode using a pre-captured CUDA graph.

        1. Pads batch to nearest power-of-2
        2. Copies actual data into static buffers
        3. Replays the captured graph
        4. Returns sliced output for actual batch size

        Args:
            forward_batch: The actual forward batch to process

        Returns:
            BaseModelOutput with logits for the actual batch size
        """
        from miniinfer.models.base import BaseModelOutput

        actual_bs = forward_batch.batch_size
        padded_bs = self._get_padded_batch_size(actual_bs)

        if padded_bs not in self.graphs:
            raise RuntimeError(
                f"No CUDA graph captured for batch_size={padded_bs}. "
                f"Available: {list(self.graphs.keys())}"
            )

        captured = self.graphs[padded_bs]

        # Build block table for the actual batch
        max_seq_len_k = int(forward_batch.seq_lens.max().item())
        actual_block_table = self.kv_cache_mgr.get_page_table(
            forward_batch, max_seq_len_k
        )
        # Convert to block indices
        actual_blocks = (
            actual_block_table[:, :: self.page_size].contiguous() // self.page_size
        ).to(torch.int32)

        # Copy actual data into static buffers
        captured.input_ids[:actual_bs].copy_(forward_batch.input_ids)
        captured.positions[:actual_bs].copy_(forward_batch.positions)
        captured.cache_seqlens[:actual_bs].copy_(forward_batch.seq_lens.to(torch.int32))
        captured.seq_lens_int64[:actual_bs].copy_(forward_batch.seq_lens)
        captured.out_cache_loc[:actual_bs].copy_(forward_batch.out_cache_loc)
        captured.req_pool_indices[:actual_bs].copy_(forward_batch.req_pool_indices)

        # Copy block table (may have fewer columns than max)
        num_blocks = actual_blocks.shape[1]
        captured.block_table[:actual_bs, :num_blocks].copy_(actual_blocks)

        # Pad remaining slots if needed
        if actual_bs < padded_bs:
            # Use values from first valid sequence for padding (safer than zeros)
            captured.input_ids[actual_bs:padded_bs] = captured.input_ids[0]
            captured.positions[actual_bs:padded_bs] = captured.positions[0]
            captured.cache_seqlens[actual_bs:padded_bs] = captured.cache_seqlens[0]
            captured.seq_lens_int64[actual_bs:padded_bs] = captured.seq_lens_int64[0]
            captured.out_cache_loc[actual_bs:padded_bs] = captured.out_cache_loc[0]
            captured.req_pool_indices[actual_bs:padded_bs] = captured.req_pool_indices[
                0
            ]
            captured.block_table[actual_bs:padded_bs] = captured.block_table[0]

        # Update metadata for attention backend with actual max_seq_len_k
        # This is a cheap operation that just updates tensor values
        self.attn_backend.update_cuda_graph_metadata(
            cache_seqlens=captured.cache_seqlens,
            max_seq_len_k=int(captured.cache_seqlens[:padded_bs].max().item()),
        )

        # Replay the captured graph
        captured.graph.replay()

        # Return sliced output for actual batch size
        return BaseModelOutput(logits=captured.output_logits[:actual_bs].clone())

    def is_available(self) -> bool:
        """Check if CUDA graphs are captured and ready for use."""
        return len(self.graphs) > 0
