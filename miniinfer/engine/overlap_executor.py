"""
OverlapExecutor - 双 Batch 交替执行器

实现 sglang 风格的 overlap scheduling：
- GPU 执行 Batch B forward 时
- CPU 并行处理 Batch A 的 sampling + 后处理

核心思路：
1. 使用独立的 forward_stream 执行 GPU 计算（非阻塞）
2. result_queue 存储待处理的 (batch, result) 元组
3. 在 GPU 计算下一 batch 时，CPU 处理上一 batch 的结果

Timeline:
    GPU forward_stream: |-- Batch A --|-- Batch B --|-- Batch C --|
    CPU main thread:    |-- idle --|-- process(A) --|-- process(B) --|
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable, Any, Deque
import torch
import logging

from miniinfer.scheduler.scheduler_batch import (
  ScheduledBatch,
  ForwardBatch,
  BatchResult,
)
from .future_map import FutureMap, FutureIndices

logger = logging.getLogger(__name__)


@dataclass
class OverlapBatchRecord:
  """记录 overlap 执行中的 batch 状态"""

  batch: ScheduledBatch
  forward_batch: ForwardBatch
  # Filled after sampling is launched.
  batch_result: Optional[BatchResult] = None
  future_indices: Optional[FutureIndices] = None
  compute_done_event: Optional[torch.cuda.Event] = None
  sample_done_event: Optional[torch.cuda.Event] = None
  copy_done_event: Optional[torch.cuda.Event] = None


class OverlapExecutor:
  """
  双 Batch 交替执行器（支持 Future Placeholder）

  管理 CUDA streams、events 和 result_queue，
  实现 GPU forward 与 CPU post-processing 的重叠执行。

  Future Placeholder 机制（SGLang 风格）:
      对于连续 decode batch，使用 "future placeholder" 打破循环依赖:
      1. schedule(N) 时，decode batch 的 input_ids 使用负数表示 "待解析 token"
      2. run_batch_async 在 forward_stream 上调用 resolve_future_input_ids() 替换占位符
      3. 这允许 schedule(N) 与 forward(N-1) 并行执行

      时序（连续 decode）:
          GPU forward_stream: |-- forward(N-1) --|-- forward(N) --|-- forward(N+1) --|
          GPU copy_stream:                       |-- copy(N-1) --|-- copy(N) --|
          CPU schedule_stream: [schedule(N)] ──► [run(N)] ──► [process(N-1)] ──► ...
                                   ↑_____ 与 forward(N-1) 真正并行 _____↑

      限制条件:
          - 仅首次 batch（无 running_batch）或 near-max-tokens 边界走保守路径
          - 连续 prefill / decode / mixed 均可 overlap

  保守路径（仅首次 batch 或 near-max-tokens 边界场景）:
      process(N-1) → schedule(N) → run_batch_async(N)
      GPU 在 schedule 期间空闲。

  Prefill Overlap:
      纯 EXTEND batch 的 input_ids 全为正数（不含 placeholder），
      但 _propagate_future_to_running_batch 会保留旧 running 请求的 future indices，
      使下一轮 can_use_placeholder=True，实现 prefill 阶段的 overlap。

  使用方式:
      executor = OverlapExecutor(max_running_requests=256)

      while True:
          # 对于连续 decode，使用 placeholder 路径
          if can_use_placeholder:
              batch = scheduler.schedule()  # 使用 placeholder input_ids
              executor.run_batch_async(batch, forward_batch, model_runner, use_placeholder=True)
              executor.process_pending_batch(process_func)  # 与 GPU 并行
          else:
              # 保守路径
              executor.process_pending_batch(process_func)
              batch = scheduler.schedule()
              executor.run_batch_async(batch, forward_batch, model_runner, use_placeholder=False)
  """

  def __init__(
    self,
    max_running_requests: int = 256,
    max_chunks_per_request: int = 4,
    device: str = "cuda",
    enable_overlap: bool = True,
  ):
    """
    初始化 OverlapExecutor

    Args:
        max_running_requests: 最大并发请求数
        max_chunks_per_request: 每个请求最多 chunk 数
        device: GPU 设备
        enable_overlap: 是否启用 overlap（可用于 A/B 测试）
    """
    self.device = device
    self.enable_overlap = enable_overlap and torch.cuda.is_available()

    if self.enable_overlap:
      # CUDA Streams
      # - forward_stream: GPU 计算流（forward + sample）
      # - copy_stream: 异步 copy 流（D2H copy）
      # - schedule_stream: 独立调度流（用于调度侧的轻量 CUDA ops/H2D copy）
      self.forward_stream = torch.cuda.Stream(device=device)
      self.copy_stream = torch.cuda.Stream(device=device)
      # Use a dedicated stream here (SGLang-style). Using the current/default
      # stream can accidentally serialize with compute on some setups.
      self.schedule_stream = torch.cuda.Stream(device=device)

      # Pinned CPU buffer for async D2H copy of next_token_ids
      # 使用 pinned memory 实现真正的异步 copy
      self.cpu_next_token_ids_buf = torch.empty(
        (max_running_requests,), dtype=torch.int64, pin_memory=True
      )

      # FutureMap: 循环 buffer 存储异步结果
      self.future_map = FutureMap(
        max_running_requests=max_running_requests,
        max_chunks_per_request=max_chunks_per_request,
        device=device,
      )

      # Result queue: 存储待处理的 (batch_record)
      self.result_queue: Deque[OverlapBatchRecord] = deque()

      # 双 buffer 保持 batch 引用存活
      # 防止 tensor 在 GPU 计算完成前被 Python GC 释放
      self.batch_record_buf = [None, None]
      self.batch_record_idx = 0

      logger.info(
        f"OverlapExecutor initialized with overlap enabled, "
        f"max_running_requests={max_running_requests}"
      )
    else:
      self.forward_stream = None
      self.copy_stream = None
      self.schedule_stream = None
      self.future_map = None
      self.cpu_next_token_ids_buf = None
      self.result_queue = deque()
      self.batch_record_buf = [None, None]
      self.batch_record_idx = 0
      logger.info("OverlapExecutor initialized with overlap disabled")

    # 追踪上一步分配的 future_indices，用于下一步 decode batch 的 placeholder
    self._last_future_indices: Optional[FutureIndices] = None
    # 追踪上一步的 batch size，用于验证
    self._last_batch_size: int = 0

  def get_placeholder_input_ids_for_decode(
    self,
    batch_size: int,
  ) -> Optional[torch.Tensor]:
    """
    获取 decode batch 应该使用的 placeholder input_ids

    使用上一步分配的 future_indices 作为 placeholder。
    只有当上一步 batch size 匹配时才能使用 placeholder。

    Args:
        batch_size: 当前 decode batch 的大小

    Returns:
        placeholder_ids 如果可用，否则 None（需要走同步路径）
    """
    if not self.enable_overlap:
      return None

    if self._last_future_indices is None:
      return None

    # 验证 batch size 匹配
    if self._last_batch_size != batch_size:
      logger.debug(
        f"Placeholder batch size mismatch: last={self._last_batch_size}, "
        f"current={batch_size}, falling back to sync path"
      )
      return None

    return self.future_map.get_placeholder_input_ids(self._last_future_indices)

  def should_disable_overlap(
    self,
    current_batch: Optional[ScheduledBatch],
  ) -> bool:
    """
    判断是否应该禁用 overlap

    禁用条件：
    1. overlap 未启用
    2. 连续两个 prefill batch（优化 TTFT）
    3. 当前 batch 为空

    Args:
        current_batch: 当前要执行的 batch
        last_batch: 上一个执行的 batch

    Returns:
        True 如果应该禁用 overlap
    """
    if not self.enable_overlap:
      return True

    if current_batch is None:
      return True

    return False

  def run_batch_async(
    self,
    batch: ScheduledBatch,
    forward_batch: ForwardBatch,
    model_runner: Any,
    use_placeholder: bool = False,
  ) -> OverlapBatchRecord:
    """
    异步执行 batch forward（非阻塞）

    在 forward_stream 上执行，立即返回，不等待完成。

    Args:
        batch: 调度的 batch
        forward_batch: 构建好的 forward batch
        model_runner: 模型执行器
        use_placeholder: input_ids 是否已设置为 placeholder（仅 decode batch）

    Returns:
        OverlapBatchRecord 包含 batch 信息和 future
    """
    # Backward-compatible wrapper: keep the old behavior (forward + sample in one call).
    # The new overlap pipeline can split compute and sampling to allow CPU-side work
    # (e.g., grammar vocab-mask generation) between them.
    record = self.run_forward_async(
      batch=batch,
      forward_batch=forward_batch,
      model_runner=model_runner,
      use_placeholder=use_placeholder,
    )
    return self.run_sample_async(record=record, model_runner=model_runner)

  def run_forward_async(
    self,
    batch: ScheduledBatch,
    forward_batch: ForwardBatch,
    model_runner: Any,
    use_placeholder: bool = False,
  ) -> OverlapBatchRecord:
    """
    Launch forward/compute on the forward stream.

    This allocates FutureMap slots for the batch and writes placeholder indices to
    `batch.output_ids` (SGLang-style), but does NOT run sampling or D2H copy.
    """
    if not self.enable_overlap:
      output = model_runner.forward(forward_batch)
      record = OverlapBatchRecord(
        batch=self._copy_batch(batch),
        forward_batch=forward_batch,
        future_indices=None,
        compute_done_event=None,
      )
      record.batch_result = BatchResult(logits=output.logits, next_token_ids=None)
      return record

    # 分配 future 槽位用于本步的输出
    bs = len(batch.reqs)
    future_indices = self.future_map.alloc_future_indices(bs)

    # sglang 风格：将 -future_indices 存入 batch.output_ids
    # 当 _filter_batch 移除完成的请求时，output_ids 也会被同步过滤，
    # 下一步 prepare_for_decode 时 input_ids = output_ids 自然是正确的 shape。
    batch.output_ids = -future_indices.indices

    # 更新追踪信息（仅用于 can_use_placeholder 的判断条件）
    self._last_future_indices = future_indices
    self._last_batch_size = bs

    # ===================================================================
    # Stream 管理（SGLang 风格）
    #
    # schedule_stream (默认流):
    #   - schedule() 调度逻辑
    #   - prepare_*() 元数据准备
    #   - forward_batch_init() 构建 ForwardBatch
    #   - 所有 CPU→GPU 异步传输（.to(device, non_blocking=True)）
    #
    # forward_stream:
    #   - model.forward() GPU 计算
    #   - model.sample() GPU 采样
    #   - resolve_future_input_ids() 替换占位符
    #
    # copy_stream:
    #   - D2H copy（next_tokens → CPU pinned buffer）
    #
    # Timeline:
    #   schedule_stream: |-- prepare(N) --|-- prepare(N+1) --|
    #   forward_stream:       |-- forward(N) --|-- forward(N+1) --|
    #   copy_stream:                   |-- copy(N) --|-- copy(N+1) --|
    # ===================================================================

    # 在 forward_stream 上执行 forward（compute）
    with torch.cuda.stream(self.forward_stream):
      # 等待 schedule_stream 完成数据准备
      self.forward_stream.wait_stream(self.schedule_stream)

      # 如果 input_ids 包含 placeholder，在 forward 前 resolve
      # 这允许 schedule(N) 与 forward(N-1) 并行
      if use_placeholder and forward_batch.input_ids is not None:
        forward_batch.input_ids = self.future_map.resolve_future_input_ids(forward_batch.input_ids)

      # GPU forward pass
      output = model_runner.forward(forward_batch)
      compute_done_event = torch.cuda.Event()
      compute_done_event.record(self.forward_stream)

    record = OverlapBatchRecord(
      batch=self._copy_batch(batch),  # 复制 batch 防止引用问题
      forward_batch=forward_batch,
      batch_result=BatchResult(logits=output.logits, next_token_ids=None),
      future_indices=future_indices,
      compute_done_event=compute_done_event,
    )
    return record

  def run_sample_async(
    self,
    record: OverlapBatchRecord,
    model_runner: Any,
    *,
    vocab_mask: Optional[torch.Tensor] = None,
  ) -> OverlapBatchRecord:
    """
    Launch sampling on the forward stream after forward/compute.

    This writes sampled next_token_ids into FutureMap, then performs an async D2H copy
    into a pinned CPU buffer on the copy stream, finally enqueuing the record for CPU
    post-processing.
    """
    if record.batch is None or record.forward_batch is None:
      raise ValueError("run_sample_async requires a valid OverlapBatchRecord")

    # Non-overlap mode: run sampling synchronously and return a ready record.
    if not self.enable_overlap:
      logits = None if record.batch_result is None else record.batch_result.logits
      if logits is None:
        raise RuntimeError("Non-overlap run_sample_async requires logits in record.batch_result")
      next_tokens = model_runner.sample(logits, record.forward_batch, vocab_mask=vocab_mask)
      record.batch_result = BatchResult(logits=logits, next_token_ids=next_tokens)
      return record

    if record.future_indices is None:
      raise RuntimeError("Overlap run_sample_async requires record.future_indices")

    bs = len(record.batch.reqs)

    # Launch sampling (and store to FutureMap) on forward_stream.
    with torch.cuda.stream(self.forward_stream):
      # Ensure any sampling metadata transfers (e.g., vocab_mask) on schedule_stream complete.
      self.forward_stream.wait_stream(self.schedule_stream)
      if record.compute_done_event is not None:
        self.forward_stream.wait_event(record.compute_done_event)

      logits = None if record.batch_result is None else record.batch_result.logits
      if logits is None:
        raise RuntimeError("run_sample_async called without logits (did run_forward_async run?)")

      next_tokens = model_runner.sample(logits, record.forward_batch, vocab_mask=vocab_mask)
      self.future_map.store_to_map(record.future_indices, next_tokens)
      sample_done_event = torch.cuda.Event()
      sample_done_event.record(self.forward_stream)

    # Async copy next_token_ids to CPU pinned memory on copy_stream.
    with torch.cuda.stream(self.copy_stream):
      self.copy_stream.wait_event(sample_done_event)
      cpu_next_tokens = self.cpu_next_token_ids_buf[:bs].copy_(next_tokens, non_blocking=True)
      copy_done_event = torch.cuda.Event()
      copy_done_event.record(self.copy_stream)

    # Update record and enqueue for CPU post-processing.
    record.sample_done_event = sample_done_event
    record.copy_done_event = copy_done_event
    record.batch_result = BatchResult(logits=logits, next_token_ids=cpu_next_tokens)

    # 保持引用存活
    self.batch_record_buf[self.batch_record_idx] = record
    self.batch_record_idx = (self.batch_record_idx + 1) % 2

    # 加入待处理队列
    self.result_queue.append(record)

    return record

  def pop_and_sync_pending(self) -> Optional[OverlapBatchRecord]:
    """
    从队列取出一个 batch 并同步 copy 事件（不调用回调）

    仅做最小 GPU 同步：等待 D2H copy 完成并 clone pinned buffer。
    返回 record 供调用方分阶段处理。

    Returns:
        OverlapBatchRecord 或 None（队列为空）
    """
    if not self.result_queue:
      return None

    record = self.result_queue.popleft()
    if record.batch_result is None:
      raise RuntimeError(
        "OverlapExecutor internal error: pending record has no batch_result. "
        "Did you enqueue it before sampling completed?"
      )

    # 等待 copy 完成（不是等待 forward 完成！）
    # copy 操作很快，这个等待通常几乎不阻塞
    if record.copy_done_event is not None:
      record.copy_done_event.synchronize()

    # 将 pinned buffer 中的数据 clone 成独立 tensor
    # 因为下一个 batch 可能覆盖 pinned buffer
    if record.batch_result.next_token_ids is not None:
      record.batch_result.next_token_ids = record.batch_result.next_token_ids.clone()

    return record

  def process_pending_batch(
    self,
    process_func: Callable[[ScheduledBatch, BatchResult], Any],
  ) -> Optional[Any]:
    """
    处理待处理队列中的下一个 batch（与 GPU 并行执行）

    从 result_queue 取出一个 batch，调用 process_func 处理。
    此时 GPU 可能正在执行下一个 batch。

    Args:
        process_func: 处理函数 (batch, result) -> outputs

    Returns:
        process_func 的返回值，或 None 如果队列为空
    """
    record = self.pop_and_sync_pending()
    if record is None:
      return None

    # 调用处理函数
    return process_func(record.batch, record.batch_result)

  def process_pending_sync(self) -> None:
    """
    同步处理所有待处理的 batch

    在禁用 overlap 场景下调用，确保所有待处理的 batch 都完成。
    """
    while self.result_queue:
      record = self.result_queue.popleft()
      if record.copy_done_event is not None:
        record.copy_done_event.synchronize()

  def has_pending(self) -> bool:
    """检查是否有待处理的 batch"""
    return len(self.result_queue) > 0

  def _copy_batch(self, batch: ScheduledBatch) -> ScheduledBatch:
    """
    浅复制 batch（保持 reqs 引用）

    复制是必要的，因为 scheduler 可能在下一轮修改 batch。
    """
    new_batch = ScheduledBatch(
      reqs=list(batch.reqs),  # 复制 list，但 Req 对象共享
      forward_mode=batch.forward_mode,
      device=batch.device,
      input_ids=batch.input_ids,
      output_ids=batch.output_ids,
      uses_placeholder=getattr(batch, "uses_placeholder", False),
      req_pool_indices=batch.req_pool_indices,
      out_cache_loc=batch.out_cache_loc,
      seq_lens=batch.seq_lens,
      seq_lens_cpu=batch.seq_lens_cpu,
      prefix_lens=batch.prefix_lens,
      extend_lens=batch.extend_lens,
      decoding_reqs=batch.decoding_reqs,
    )
    return new_batch

  def synchronize(self) -> None:
    """同步所有 CUDA 流"""
    if self.forward_stream is not None:
      self.forward_stream.synchronize()

  def reset(self) -> None:
    """重置执行器状态"""
    self.result_queue.clear()
    self.batch_record_buf = [None, None]
    self.batch_record_idx = 0
    if self.future_map is not None:
      self.future_map.reset()
