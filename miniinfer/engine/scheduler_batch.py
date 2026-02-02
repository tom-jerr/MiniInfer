from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, List
from layers.attention_backend.base_backend import AttentionBackend
import torch
from itertools import count
from copy import copy
from miniinfer.utils.sampling_params import SamplingParams


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

    def is_extend(self) -> bool:
        return self == ForwardMode.EXTEND

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE


class BatchType(IntEnum):
    """Batch 类型"""

    PREFILL_ONLY = auto()
    DECODE_ONLY = auto()
    MIXED = auto()


class Req:
    # block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.req_id = next(Req.counter)
        self.origin_input_ids = copy(
            token_ids
        )  # 切断与外部变量的联系，让 Req 拥有这份数据的独占权
        self.output_ids = []
        # fill_ids = origin_input_ids + output_ids. Used in chunked prefill.
        self.fill_ids = []
        # keep original sampling params for downstream components (e.g., sampler)
        self.sampling_params = sampling_params
        # ============ sampling related ============
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p

        # ============ finish related ============
        self.is_retracted = False
        self.finished = False
        self.finished_reason = ""

        # ============ kv cache related ============
        self.req_pool_idx: int = (
            -1
        )  # The index in the request pool, for radix cache and kv cache management
        self.extend_input_len = 0
        self.prefix_indices: torch.Tensor = None
        # ============ radix cache related ============
        """used last_node and cache_protected_len in update radix cache"""
        self.last_node = None
        self.cache_protected_len = 0  # 已经保护的 KV cache 长度，防止被驱逐

        # ============ chunked prefill related ============
        self.is_chunked = False  # 是否是 chunked 请求
        self.chunked_prefill_len = 0  # 当前 chunk 已经处理的长度
        self.total_input_len = len(token_ids)  # 总输入长度

    @property
    def remaining_prefill_len(self) -> int:
        """剩余需要 prefill 的长度"""
        return (
            self.total_input_len - self.chunked_prefill_len - self.cache_protected_len
        )

    def is_prefill_complete(self) -> bool:
        """检查 prefill 是否完成"""
        return (
            self.chunked_prefill_len + self.cache_protected_len >= self.total_input_len
        )


class ChunkedReq:
    """
    Chunked Prefill 请求的状态追踪

    当一个请求太大无法在一个 batch 中完成 prefill 时，
    会被拆分成多个 chunk 处理。这个类用于追踪 chunked 状态。

    注意：ChunkedReq 不会进入 decode 阶段，直到所有 chunk 都处理完成。
    """

    def __init__(
        self,
        req: Req,
        cached_len: int,
        chunk_size: int,
    ):
        """
        初始化 ChunkedReq

        Args:
            req: 原始请求
            cached_len: 已缓存的长度（prefix cache 命中）
            chunk_size: 当前 chunk 的大小
        """
        self.req = req
        self.cached_len = cached_len  # 包括 prefix cache + 已处理的 chunk
        self.chunk_size = chunk_size  # 当前 chunk 要处理的 token 数

        # 更新原始请求的状态
        req.is_chunked = True

    @property
    def input_len(self) -> int:
        """原始输入总长度"""
        return self.req.total_input_len

    @property
    def remaining_len(self) -> int:
        """处理完当前 chunk 后，还剩余的长度"""
        return self.input_len - self.cached_len - self.chunk_size

    @property
    def is_last_chunk(self) -> bool:
        """是否是最后一个 chunk"""
        return self.remaining_len <= 0

    def get_chunk_input_ids(self) -> list[int]:
        """获取当前 chunk 的 input_ids"""
        start = self.cached_len
        end = start + self.chunk_size
        return self.req.origin_input_ids[start:end]

    def update_after_chunk(self):
        """
        处理完当前 chunk 后更新状态

        如果还有剩余，更新 cached_len 以便下一个 chunk
        """
        self.req.chunked_prefill_len = self.cached_len + self.chunk_size
        if self.is_last_chunk:
            self.req.is_chunked = False  # 完成所有 chunk


@dataclass
class ScheduledBatch:
    reqs: List[Req]
    forward_mode: ForwardMode = ForwardMode.IDLE
    device: torch.device = None
    # ============ model forward related ============
    input_ids: torch.Tensor = None
    output_ids: torch.Tensor = None

    # =========== kv cache related ============
    req_pool_indices: torch.Tensor = None
    out_cache_loc: torch.Tensor = None

    # ============ some metadata ============
    seq_lens: torch.Tensor = (
        None  # from req.fill_ids = origin_input_ids + output_ids, for forward batch k_cache len
    )
    seq_lens_cpu: Optional[torch.Tensor] = None

    # ======== extend related ========
    prefix_lens: List[int] = None
    extend_lens: List[int] = None

    # ======== chunked prefill related ========
    decoding_reqs: List[Req] = None

    @classmethod
    def init_new(cls, reqs: List[Req], device: torch.device):
        return cls(
            reqs=reqs,
            device=device,
        )

    def debug_metadata(self):
        print("ScheduledBatch metadata:")
        print(f"  forward_mode: {self.forward_mode}")
        print(f"  input_ids: {self.input_ids}")
        print(f"  output_ids: {self.output_ids}")
        print(f"  req_pool_indices: {self.req_pool_indices}")
        print(f"  out_cache_loc: {self.out_cache_loc}")
        print(f"  seq_lens: {self.seq_lens}")
        print(f"  prefix_lens: {self.prefix_lens}")
        print(f"  extend_lens: {self.extend_lens}")


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
    req_pool_indices: torch.Tensor = None
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor = None
    attn_backend: AttentionBackend = None
    # sequences that participate in this forward step (for sampling metadata)
    all_seqs: List[Req] = None
    # ============ some metadata for k ============
    seq_lens: torch.Tensor = None

    # =========== TODO: Maybe remove ===========
    seq_lens_sum: int = 0
    seq_lens_cpu: Optional[torch.Tensor] = None

    # ============ some metadata for q
    # extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    # extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None

    @classmethod
    def init_new(cls, batch: ScheduledBatch, attn_backend: AttentionBackend):
        forward_batch = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.reqs),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            out_cache_loc=batch.out_cache_loc,
            seq_lens=batch.seq_lens,
            seq_lens_cpu=batch.seq_lens_cpu,
            attn_backend=attn_backend,
            all_seqs=batch.reqs,
        )

        if batch.forward_mode.is_extend():
            # forward_batch.extend_num_tokens = batch.input_ids.shape[0]
            forward_batch.extend_seq_lens = batch.extend_lens
            forward_batch.extend_prefix_lens = torch.tensor(
                batch.prefix_lens, dtype=torch.int64
            ).to(batch.device)
            forward_batch.extend_prefix_lens_cpu = batch.prefix_lens
            forward_batch.extend_seq_lens_cpu = batch.extend_lens
            # positions and start loc for extend
            forward_batch.positions = compute_position_torch(
                forward_batch.extend_prefix_lens, forward_batch.extend_seq_lens
            )
        else:
            forward_batch.positions = clamp_position(batch.seq_lens)

        return forward_batch

    def debug_metadata(self):
        print("ForwardBatch metadata:")
        print(f"  forward_mode: {self.forward_mode}")
        print(f"  attn_backend type: {self.attn_backend.type()}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  input_ids: {self.input_ids}")
        print(f"  req_pool_indices: {self.req_pool_indices}")
        print(f"  out_cache_loc: {self.out_cache_loc}")
        print(f"  seq_lens: {self.seq_lens}")
        print(f"  extend_seq_lens: {self.extend_seq_lens}")
        print(f"  extend_prefix_lens: {self.extend_prefix_lens}")


@dataclass
class BatchResult:
    logits: torch.Tensor
    next_token_ids: torch.Tensor


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.cat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    # extend_start_loc = torch.zeros_like(extend_seq_lens)
    # extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64)


@torch.compile(dynamic=True)
def clamp_position(seq_lens):
    return torch.clamp((seq_lens - 1), min=0).to(torch.int64)
