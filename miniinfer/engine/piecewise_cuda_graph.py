"""Piecewise CUDA graph for prefill（MoE-extensible）。

背景：prefill 走 varlen attention，total_tokens 随 batch 变化，无法像 decode 那样
按 power-of-2 batch size 直接捕获单图。vLLM 的 cuDAG 用 **piecewise** 思路：把
forward 拆成若干 segment，**静态形状的 segment（GEMM/norm/embedding）用 cuda
graph 捕获**，**动态形状的 segment（varlen attention、未来的 MoE router/grouped
GEMM）eager 执行**，二者通过预分配 buffer 串联。

可扩展性（MoE）：
- ``Segment`` 注册机制：每个 segment 声明是否 ``eager``（动态，不捕获）。
- dense 模型：1 个 captured segment（整层 forward，v1 用「按 token bucket 全量
  捕获 + dummy padding seq」让 varlen attention 也能进图）。
- MoE 模型（后续）：把 MoE 的 router + grouped GEMM 注册为 eager segment，前后的
  dense GEMM 仍 captured —— 框架按注册顺序串联，无需改 runner。

v1（dense，本提交）：
- 按 token bucket（powers of 2 ≤ max_extend_len）捕获整层 forward。
- 用 dummy padding sequence 把 total_tokens 补到 bucket：dummy token 的
  out_cache_loc 指向保留的 page 0，KV 写入 page 0（覆盖，无害）；varlen attention
  处理 [actual + dummy] = [bucket]，输出 [bucket]，末尾取 [:actual]。
- 静态 buffer：input_ids/positions/out_cache_loc [bucket]、cu_seqlens_q/k
  [max_nseq+2]、seq_lens/extend_seq_lens [max_nseq] 等，replay 前 in-place 更新。

注意：v1 仅支持 **无 prefix-cache 命中** 的 prefill（q/k/v 即 extend tokens，
  路径简单）。命中 prefix cache 时回落 eager（forward_extend 原路径）。MoE 接入
  时再按 segment 拆分。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from miniinfer.utils import get_logger

logger = get_logger(__name__)


@dataclass
class Segment:
  """一个 forward segment。

  Attributes:
    name: 段名。
    eager: True=动态形状，每步 eager 执行（varlen attention / MoE router）；
      False=静态形状，按 bucket 捕获 cuda graph。
    run: 执行函数 ``run(ctx) -> None``，读写 ctx 中的 buffer。
  """

  name: str
  eager: bool
  run: Callable[["ReplayContext"], None]


@dataclass
class ReplayContext:
  """replay 时各 segment 共享的上下文（buffer + 元信息）。"""

  bucket: int
  actual_tokens: int
  # 静态 buffer（replay 前 in-place 更新内容）
  input_ids: torch.Tensor
  positions: torch.Tensor
  out_cache_loc: torch.Tensor
  # 由 attn_backend 的 extend cuda-graph metadata 提供
  attn_metadata: Any = None
  # 自定义槽位（MoE 等 segment 可挂载中间 buffer）
  slots: Dict[str, Any] = field(default_factory=dict)


class PiecewiseCudaGraphRunner:
  """Prefill 的 piecewise cuda graph runner（MoE-extensible）。

  用法：
    runner = PiecewiseCudaGraphRunner(model_runner, buckets=[1024,2048,4096,8192])
    runner.register_segment(Segment("forward", eager=False, run=...))
    runner.capture()           # 每个 bucket 捕获
    out = runner.replay(fb)    # prefill 时调用
  """

  def __init__(
    self,
    model_runner: Any,
    buckets: Optional[List[int]] = None,
    max_nseq: int = 256,
  ):
    self.model_runner = model_runner
    self.device = model_runner.device
    self.dtype = model_runner.config.dtype
    self.kv_cache_mgr = model_runner.kv_cache_mgr
    self.attn_backend = model_runner.attn_backend
    self.model = model_runner.model
    self.max_nseq = max_nseq
    cfg = model_runner.model_config
    self.vocab_size = cfg.vocab_size
    self.hidden_size = cfg.hidden_size
    self.page_size = self.kv_cache_mgr.page_size
    # token bucket（powers of 2 ≤ max_extend_len）
    max_extend = getattr(model_runner.config, "max_extend_len", 8192)
    if buckets is None:
      buckets = []
      b = 1024
      while b <= max_extend:
        buckets.append(b)
        b *= 2
    self.buckets = buckets
    # segment 注册（顺序执行）
    self.segments: List[Segment] = []
    # 每个 bucket 的捕获产物：{bucket: {"graph":..., "buffers":..., "output_logits":...}}
    self.graphs: Dict[int, dict] = {}
    self._captured = False

  # ------------------------------------------------------------------
  # segment 注册（MoE 扩展点）
  # ------------------------------------------------------------------
  def register_segment(self, seg: Segment) -> None:
    self.segments.append(seg)

  def is_available(self) -> bool:
    return self._captured and len(self.graphs) > 0

  def _bucket_for(self, total_tokens: int) -> Optional[int]:
    for b in self.buckets:
      if total_tokens <= b:
        return b
    return None  # 超过最大 bucket，回落 eager

  # ------------------------------------------------------------------
  # capture
  # ------------------------------------------------------------------
  def capture(self) -> None:
    from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode

    for bucket in self.buckets:
      try:
        self._capture_bucket(bucket, ForwardBatch, ForwardMode)
      except Exception as e:
        logger.warning(f"PiecewiseCudaGraph: capture bucket={bucket} failed: {e!r}; skip")
    self._captured = True
    logger.info(
      f"PiecewiseCudaGraph captured buckets={list(self.graphs.keys())} "
      f"(segments={[s.name for s in self.segments]})"
    )

  def _capture_bucket(self, bucket: int, ForwardBatch, ForwardMode) -> None:
    device = self.device
    nseq = 2  # 捕获用 2 个 dummy seq（1 actual + 1 dummy padding）
    # 静态 buffer
    input_ids = torch.zeros(bucket, dtype=torch.int64, device=device)
    positions = torch.zeros(bucket, dtype=torch.int64, device=device)
    # out_cache_loc：actual 用 page 0 slot，dummy 也用 page 0（保留页，覆盖无害）
    out_cache_loc = torch.zeros(bucket, dtype=torch.int64, device=device)
    seq_lens = torch.full((nseq,), bucket, dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor([1, bucket - 1], dtype=torch.int64, device=device)
    extend_prefix_lens = torch.zeros(nseq, dtype=torch.int64, device=device)
    req_pool_indices = torch.zeros(nseq, dtype=torch.int64, device=device)

    fb = ForwardBatch(
      forward_mode=ForwardMode.EXTEND,
      batch_size=nseq,
      input_ids=input_ids,
      positions=positions,
      req_pool_indices=req_pool_indices,
      out_cache_loc=out_cache_loc,
      seq_lens=seq_lens,
      seq_lens_int32=seq_lens.to(torch.int32),
      all_seqs=None,
      attn_backend=self.attn_backend,
    )
    fb.extend_seq_lens = extend_seq_lens
    fb.extend_prefix_lens = extend_prefix_lens
    fb.extend_seq_lens_cpu = extend_seq_lens.tolist()
    fb.extend_prefix_lens_cpu = extend_prefix_lens.tolist()
    fb.max_seq_len = bucket
    fb.positions = positions

    # 让 attn backend 建立 extend 的 cuda-graph metadata（静态 cu_seqlens 等）。
    self.attn_backend.init_extend_cuda_graph_metadata(
      forward_batch=fb,
      bucket=bucket,
      max_nseq=self.max_nseq,
    )

    output_logits = torch.empty(
      (bucket, self.vocab_size), dtype=self.dtype, device=device
    )

    # warmup
    torch.cuda.synchronize()
    with torch.no_grad():
      out = self.model.forward(input_ids, positions, fb, return_hidden_states=False)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
      out = self.model.forward(input_ids, positions, fb, return_hidden_states=False)
      output_logits.copy_(out.logits)

    self.graphs[bucket] = {
      "graph": graph,
      "input_ids": input_ids,
      "positions": positions,
      "out_cache_loc": out_cache_loc,
      "seq_lens": seq_lens,
      "extend_seq_lens": extend_seq_lens,
      "extend_prefix_lens": extend_prefix_lens,
      "req_pool_indices": req_pool_indices,
      "output_logits": output_logits,
    }
    logger.debug(f"PiecewiseCudaGraph captured bucket={bucket}")

  # ------------------------------------------------------------------
  # replay
  # ------------------------------------------------------------------
  def replay(self, forward_batch) -> Any:
    """prefill replay。返回 BaseModelOutput（logits 已取 [:actual_tokens]）。"""
    from miniinfer.models.base import BaseModelOutput

    # 仅支持无 prefix-cache 命中的 prefill（q/k/v = extend tokens）。
    if forward_batch.extend_prefix_lens_cpu and any(
      p > 0 for p in forward_batch.extend_prefix_lens_cpu
    ):
      return None  # 命中 prefix cache，回落 eager

    total_tokens = forward_batch.input_ids.shape[0]
    bucket = self._bucket_for(total_tokens)
    if bucket is None:
      return None  # 超最大 bucket，回落 eager

    g = self.graphs[bucket]
    nseq = forward_batch.batch_size
    actual = total_tokens

    # 1. 拷贝 actual token 数据到静态 buffer
    g["input_ids"][:actual].copy_(forward_batch.input_ids)
    g["positions"][:actual].copy_(forward_batch.positions)
    g["out_cache_loc"][:actual].copy_(forward_batch.out_cache_loc)
    # dummy padding token（[actual:bucket]）→ page 0（保留页），input_ids=0
    if actual < bucket:
      g["input_ids"][actual:].zero_()
      g["out_cache_loc"][actual:].zero_()
      # dummy positions 用 0（不参与实际 attention 结果，会被丢弃）
      g["positions"][actual:].zero_()

    # 2. 更新 attn extend cuda-graph metadata（actual seqs + 1 dummy padding seq）
    self.attn_backend.update_extend_cuda_graph_metadata(
      forward_batch=forward_batch,
      bucket=bucket,
      static_buffers=g,
    )

    # 3. replay
    g["graph"].replay()

    # 4. 取 actual 部分
    logits = g["output_logits"][:actual].clone()
    return BaseModelOutput(logits=logits, last_hidden_state=None, hidden_states=None)
