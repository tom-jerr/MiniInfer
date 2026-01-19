from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, List
import torch


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()


@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode

    # ============ model forward related ============
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # Position information
    positions: torch.Tensor = None

    # ============ kv cache related ============
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # ============ some metadata ============
    # The sequence length
    seq_lens: torch.Tensor
    # The sum of all sequence lengths
    seq_lens_sum: int
    # Optional seq_lens on cpu
    seq_lens_cpu: Optional[torch.Tensor] = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
