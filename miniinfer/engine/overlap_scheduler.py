"""
OverlapScheduler - 双 Stream Overlap 调度器

参考 mini-sglang 的 Scheduler.overlap_loop() 实现。

核心设计:
- 2 个 CUDA Stream: schedule_stream + compute_stream
- schedule_stream: 调度 + metadata 准备 + H2D 传输 + D2H 传输
- compute_stream: 纯 GPU 计算（forward + sample + token_pool writeback）

Timeline:
    schedule_stream: |D2H(N-1)+process(N-1)|--schedule(N)+H2D--|   |D2H(N)+process(N)|
    compute_stream:  |---forward(N-1)+sample+writeback---------|---forward(N)+sample+writeback---|

关键优化:
- D2H 放在 schedule_stream 而不是 compute_stream，
  避免 D2H 阻塞下一批 forward 启动
- input_ids 通过 token_pool GPU gather 获取，无需 H2D（decode 阶段）
- 采样后 token_pool writeback 在 compute_stream 上
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, NamedTuple, Tuple, Any, Callable

import torch
import logging

from miniinfer.engine.token_pool import TokenPool
from miniinfer.scheduler.scheduler_batch import ForwardBatch

if TYPE_CHECKING:
  from miniinfer.scheduler.scheduler_batch import ScheduledBatch
  from miniinfer.engine.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class ForwardInput(NamedTuple):
  """Overlap forward 的输入"""

  batch: "ScheduledBatch"
  # (table_indices, positions) 用于 token_pool.gather_input
  input_mapping: Tuple[torch.Tensor, torch.Tensor]
  # (table_indices, positions) 用于 token_pool.write_output
  write_mapping: Tuple[torch.Tensor, torch.Tensor]


@dataclass
class ForwardOutput:
  """Overlap forward 的输出"""

  next_tokens_gpu: torch.Tensor  # GPU tensor, shape=(batch_size,)
  sample_done_event: torch.cuda.Event  # compute_stream 上 sample 完成的 event
  # forward 时的 batch_size，用于 D2H 复制（避免 batch.reqs 被 _filter_batch 修改后长度不一致）
  batch_size: int = 0
  # CPU tensor 会在 schedule_stream 上 D2H 后填充
  next_tokens_cpu: Optional[torch.Tensor] = None
  copy_done_event: Optional[torch.cuda.Event] = None


@dataclass
class ForwardData:
  """一次 forward 的完整数据，用于 process_last_data"""

  forward_input: ForwardInput
  forward_output: ForwardOutput


class OverlapScheduler:
  """
  双 Stream Overlap 调度器

  管理 schedule_stream 和 compute_stream，
  实现 GPU forward 与 CPU 处理的真正重叠。
  """

  def __init__(
    self,
    token_pool: TokenPool,
    model_runner: "ModelRunner",
    max_running_requests: int = 256,
    device: str = "cuda",
  ):
    """
    初始化 OverlapScheduler

    Args:
        token_pool: GPU 常驻 token 存储
        model_runner: 模型执行器
        max_running_requests: 最大并发请求数
        device: GPU 设备
    """
    self.token_pool = token_pool
    self.model_runner = model_runner
    self.device = device

    # 双 Stream 架构
    self.schedule_stream = torch.cuda.Stream(device=device)
    self.compute_stream = torch.cuda.Stream(device=device)

    # 设置默认 stream 为 schedule_stream（调度相关操作）
    # compute_stream 用于纯 GPU 计算

    # Pinned CPU buffer for async D2H copy
    self.cpu_next_tokens_buf = torch.empty(
      (max_running_requests,), dtype=torch.int32, pin_memory=True
    )

    # 复用的 CUDA Events
    self._sample_done_event = torch.cuda.Event()
    self._copy_done_event = torch.cuda.Event()

    logger.info(
      f"OverlapScheduler initialized: device={device}, "
      f"schedule_stream={self.schedule_stream}, compute_stream={self.compute_stream}"
    )

  def run_forward(self, forward_input: ForwardInput) -> ForwardOutput:
    """
    在 compute_stream 上执行 forward + sample + writeback

    这是纯 GPU 操作，不含任何 D2H。

    Args:
        forward_input: 包含 batch 和 mapping 信息

    Returns:
        ForwardOutput: 包含 next_tokens_gpu 和 sample_done_event
    """
    batch = forward_input.batch
    input_mapping = forward_input.input_mapping
    write_mapping = forward_input.write_mapping

    with torch.cuda.stream(self.compute_stream):
      # 等待 schedule_stream 完成数据准备（H2D 传输）
      self.compute_stream.wait_stream(self.schedule_stream)

      # ======== Step 1: GPU gather input_ids ========
      # 对于 decode: input_ids = token_pool[table_indices, device_len - 1]
      # 对于 extend: input_ids = token_pool[table_indices, [cached_len, ..., device_len-1]]
      batch.input_ids = self.token_pool.gather_input(
        table_indices=input_mapping[0],
        positions=input_mapping[1],
      )

      # ======== Step 2: Model forward ========
      # 构建 ForwardBatch 并执行 forward
      forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)
      output = self.model_runner.forward(forward_batch)
      logits = output.logits

      # ======== Step 3: Sample ========
      next_tokens = self.model_runner.sample(logits, forward_batch)

      # ======== Step 4: Token pool writeback ========
      # 只写入 can_decode=True 的请求
      # write_mapping[1] 中 -1 表示不写入
      write_table_indices = write_mapping[0]
      write_positions = write_mapping[1]
      valid_mask = write_positions >= 0
      if valid_mask.any():
        self.token_pool.write_output(
          table_indices=write_table_indices[valid_mask],
          positions=write_positions[valid_mask],
          next_tokens=next_tokens[valid_mask],
        )

      # ======== Step 5: Record sample_done_event ========
      # D2H 将在下一轮 schedule_stream 上执行
      sample_done_event = torch.cuda.Event()
      sample_done_event.record(self.compute_stream)

      # 保存 forward 时的 batch_size（用于 D2H 复制，避免 batch.reqs 被修改后长度不一致）
      bs_at_forward = len(batch.reqs)

      # 更新 req 状态
      for req in batch.reqs:
        if hasattr(req, "complete_one"):
          req.complete_one()

    return ForwardOutput(
      next_tokens_gpu=next_tokens,
      sample_done_event=sample_done_event,
      batch_size=bs_at_forward,
    )

  def process_last_data(
    self,
    last_data: Optional[ForwardData],
    process_fn: Callable[["ScheduledBatch", torch.Tensor], Any],
  ) -> Optional[Any]:
    """
    处理上一轮的 forward 结果

    在 schedule_stream 上执行 D2H，然后 CPU 处理结果。
    此时 compute_stream 可能正在执行当前轮的 forward。

    Args:
        last_data: 上一轮的 forward 数据
        process_fn: CPU 处理函数 (batch, next_tokens_cpu) -> Any

    Returns:
        process_fn 的返回值
    """
    if last_data is None:
      return None

    batch = last_data.forward_input.batch
    forward_output = last_data.forward_output

    # ======== Step 1: D2H on schedule_stream ========
    with torch.cuda.stream(self.schedule_stream):
      # 等待 compute_stream 上的 sample 完成
      self.schedule_stream.wait_event(forward_output.sample_done_event)

      # 使用 forward 时保存的 batch_size，而非 len(batch.reqs)
      # 因为 batch.reqs 可能已被 _filter_batch() 修改（连续 decode 时 batch == running_batch）
      bs = forward_output.batch_size

      # Async D2H copy to pinned buffer
      cpu_next_tokens = self.cpu_next_tokens_buf[:bs].copy_(
        forward_output.next_tokens_gpu, non_blocking=True
      )

      # Record copy_done_event
      copy_done_event = torch.cuda.Event()
      copy_done_event.record(self.schedule_stream)

    # ======== Step 2: Wait for D2H to complete ========
    # D2H 很快，通常几乎不阻塞
    copy_done_event.synchronize()

    # ======== Step 3: CPU processing ========
    # Clone to avoid pinned buffer being overwritten by next batch
    next_tokens_cpu = cpu_next_tokens.clone()

    return process_fn(batch, next_tokens_cpu)

  def overlap_step(
    self,
    last_data: Optional[ForwardData],
    schedule_fn: Callable[[], Optional[ForwardInput]],
    process_fn: Callable[["ScheduledBatch", torch.Tensor], Any],
  ) -> Tuple[Optional[ForwardData], Optional[Any]]:
    """
    执行一步 overlap 循环

    Timeline:
      1. schedule_stream: D2H(N-1) + process(N-1)  [与 compute 残留工作并行]
      2. schedule_stream: schedule(N) + H2D       [CPU 调度 + metadata 准备]
      3. compute_stream: forward(N) + sample + writeback

    Args:
        last_data: 上一轮的 forward 数据（如果有）
        schedule_fn: 调度函数，返回 ForwardInput
        process_fn: CPU 处理函数

    Returns:
        (ongoing_data, process_result):
        - ongoing_data: 本轮的 ForwardData，用于下一轮处理
        - process_result: process_fn 的返回值
    """
    # ======== Phase 1: Process last batch results ========
    process_result = self.process_last_data(last_data, process_fn)

    # ======== Phase 2: Schedule next batch ========
    with torch.cuda.stream(self.schedule_stream):
      forward_input = schedule_fn()

    # ======== Phase 3: Run forward on compute_stream ========
    ongoing_data = None
    if forward_input is not None:
      forward_output = self.run_forward(forward_input)
      ongoing_data = ForwardData(
        forward_input=forward_input,
        forward_output=forward_output,
      )

    return ongoing_data, process_result

  def synchronize(self) -> None:
    """同步两个 stream"""
    self.schedule_stream.synchronize()
    self.compute_stream.synchronize()

  def reset(self) -> None:
    """重置状态"""
    self.synchronize()
