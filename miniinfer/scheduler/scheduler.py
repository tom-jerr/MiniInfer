"""
Scheduler - 单机版调度器

负责:
1. 管理请求队列 (waiting, running, finished)
2. 组织 batch (prefill-only / decode-only / mixed)
3. 调用 KVCacheManager 分配 KV Cache
4. 增量流式解码
5. Token 预算控制，防止 OOM
6. 支持 Chunked Prefill
7. 防止 decode 饿死 + 动态 new_token_ratio 调节

调度策略 (SGLang 风格):
    主循环先尝试拼一个新的 prefill 批次 (get_new_batch_prefill)；
    只有当 prefill 批次为空时，才回退到现有的 decode 批次，
    因此默认是 "prefill 优先、decode 兜底"。
    为防止 decode 饿死和 OOM:
    - PrefillAdder 初始化时从可用 token 数中扣除 decode 请求的 new_token 预留
    - Decode 阶段通过 check_decode_mem 检查是否需要收缩批次
    - 连续内存不足时动态调高 new_token_ratio，给 decode 更多预留空间
    - 全部预算以 page_size 粒度对齐
"""

from typing import List, Optional, Any
import logging
import torch

from miniinfer.config.engine.config import EngineConfig
from miniinfer.kvcache.kv_cache_manager import KVCacheManager
from miniinfer.utils.profiler_utils import profile_methods
from miniinfer.scheduler.prefill_adder import PrefillAdder, AddReqResult
from miniinfer.scheduler.scheduler_batch import (
  Req,
  ChunkedReq,
  ScheduledBatch,
  ForwardBatch,
  ForwardMode,
  BatchResult,
)

logger = logging.getLogger(__name__)

# ======================= 常量 =======================
# new_token_ratio 的初始/最小/最大值
_INIT_NEW_TOKEN_RATIO = 0.3
_MIN_NEW_TOKEN_RATIO = 0.1
_MAX_NEW_TOKEN_RATIO = 0.85
# new_token_ratio 每次调节的步长
_NTR_GROW_STEP = 0.08
_NTR_DECAY_STEP = 0.02
# decode 连续 OOM 收缩次数达到此值后开始调高 new_token_ratio
_RETRACT_THRESHOLD = 2


@profile_methods("Scheduler")
class Scheduler:
  """
  调度器 - 防 decode 饿死 + 防 OOM

  核心机制:
  1. new_token_ratio 动态调节:
     - 初始 0.3，每次 decode 内存不足 (retract) 时步进增大
     - 连续若干轮 decode 顺利时步进缩小，释放更多 prefill 空间
  2. PrefillAdder 预算:
     - rem_total_tokens = available_kv - decode_reserve (page 对齐)
     - rem_input_tokens = min(rem_total_tokens, max_extend_len) (page 对齐)
  3. Decode 内存检查 (check_decode_mem):
     - 估算下一步所有 decode 请求需要的 page 数
     - 不足时从 batch 尾部收缩请求 (retract)，被收缩的请求回到 waiting 队列
  """

  def __init__(
    self,
    config: EngineConfig,
    tokenizer: Any,
    kv_cache_mgr: KVCacheManager,
  ):
    self.engine_config = config
    self.tokenizer = tokenizer
    self.kv_cache_mgr = kv_cache_mgr
    self.max_batch_size = config.max_num_seqs
    self.max_extend_len = getattr(config, "max_extend_len", 8192)
    self.page_size = getattr(config, "page_size", 256)
    self.gpu_memory_utilization = getattr(config, "gpu_memory_utilization", 0.6)

    self.enable_chunked_prefill = getattr(config, "enable_chunked_prefill", False)
    self.chunked_prefill_size = getattr(config, "chunked_prefill_size", 4096)

    # ============ 动态 new_token_ratio ============
    self.new_token_ratio = _INIT_NEW_TOKEN_RATIO
    self.num_continuous_retract = 0  # 连续收缩次数
    self.num_continuous_decode_ok = 0  # 连续 decode 顺利次数

    # ============ 请求队列 ============
    self.waiting_queue: List[Req] = []
    self.running_batch: ScheduledBatch = ScheduledBatch(reqs=[])
    self.chunking_reqs: List[ChunkedReq] = []
    self.cur_batch: Optional[ScheduledBatch] = None
    self.last_batch: Optional[ScheduledBatch] = None
    self.num_retracted_reqs: int = 0

    # ============ 已完成的请求 ============
    self.finished_reqs: List[Req] = []
    # Finished reqs whose KV/prefix-cache release is deferred to overlap with compute.
    self.pending_release_reqs: List[Req] = []

    # EOS token id
    self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    logger.info(
      f"Scheduler initialized: max_batch_size={self.max_batch_size}, "
      f"max_extend_len={self.max_extend_len}, "
      f"page_size={self.page_size}, "
      f"enable_chunked_prefill={self.enable_chunked_prefill}, "
      f"new_token_ratio={self.new_token_ratio}"
    )

  # ======================== 公共 API ========================

  def step(self):
    batch = self.schedule()
    self.cur_batch = batch
    if batch is None:
      return []

    result = self.run_batch(batch)
    output_texts = self.process_batch_result(batch, result)
    return output_texts

  def add(self, req: Req):
    """添加新请求到等待队列"""
    self.waiting_queue.append(req)

  def has_unfinished(self) -> bool:
    return (
      len(self.waiting_queue) > 0 or len(self.running_batch.reqs) > 0 or len(self.chunking_reqs) > 0
    )

  def get_num_unfinished(self) -> int:
    return len(self.waiting_queue) + len(self.running_batch.reqs) + len(self.chunking_reqs)

  # ======================== 调度主循环 ========================

  def schedule(
    self,
    device: torch.device = None,
    *,
    skip_decode_input_ids: bool = False,
  ) -> Optional[ScheduledBatch]:
    """
    调度主循环 — Sarathi-Serve 风格 stall-free batching

    当开启 chunked prefill 时:
    1. 过滤已完成请求
    2. 尝试构建 mixed batch (decode + prefill)，decode 优先
    3. 如果 prefill 队列为空且有 decode → 走纯 decode 路径

    当未开启 chunked prefill 时:
    1. 过滤已完成请求
    2. 尝试 prefill 批次
    3. 如果 prefill 为空 → 走 decode 路径
    """
    # 打印当前调度器状态（空闲页数等）
    if logger.isEnabledFor(logging.DEBUG):
      self._print_scheduler_info()

    # Step 1: 过滤已完成的请求
    self._filter_batch(self.running_batch)
    running_bs = len(self.running_batch.reqs)

    # Step 2: 尝试拼 prefill/mixed 批次
    prefill_batch = self._get_new_batch_prefill(device, skip_decode_input_ids=skip_decode_input_ids)

    if prefill_batch is not None:
      if logger.isEnabledFor(logging.DEBUG):
        self._print_batch_info(prefill_batch, "PREFILL/MIXED")
      return prefill_batch

    # Step 3: 没有 prefill → 回退到纯 decode
    if running_bs > 0:
      decode_batch = self._get_decode_batch(device, skip_decode_input_ids=skip_decode_input_ids)
      if decode_batch is not None:
        if logger.isEnabledFor(logging.DEBUG):
          self._print_batch_info(decode_batch, "DECODE")
      return decode_batch

    logger.debug("[Scheduler] No batch to execute this round")
    return None

  # ======================== Prefill 批次构建 ========================

  def _get_new_batch_prefill(
    self, device: torch.device, *, skip_decode_input_ids: bool = False
  ) -> Optional[ScheduledBatch]:
    """
    尝试构建一个 prefill 批次

    预算计算:
    1. available_kv = kv_cache_mgr.available_tokens()
    2. PrefillAdder 内部从 available_kv 中扣除 decode 预留
    3. 先处理 chunking_reqs 中尚未完成的 chunk
    4. 再从 waiting_queue 中取新请求
    """
    running_reqs = self.running_batch.reqs

    if not self.waiting_queue and not self.chunking_reqs:
      return None

    available_kv = self.kv_cache_mgr.available_tokens()

    # 创建 PrefillAdder —— 核心预算控制
    # 精确传入 running_reqs，按每个请求的 min(剩余输出, CLIP) * ratio 预留
    # BUG FIX: 在 mixed batch 场景下，need to account for running batch size
    # 避免 total batch size 超过 request pool 容量
    effective_max_batch_size = self.max_batch_size - len(running_reqs)
    adder = PrefillAdder(
      available_kv_tokens=available_kv,
      running_reqs=running_reqs,
      new_token_ratio=self.new_token_ratio,
      max_extend_len=self.max_extend_len,
      max_batch_size=effective_max_batch_size,
      chunk_size=(self.chunked_prefill_size if self.enable_chunked_prefill else None),
      page_size=self.page_size,
    )

    # 如果预算不足，直接返回
    if adder.no_remaining_budget():
      logger.debug(f"No budget for prefill: available_kv={available_kv}")
      return None

    # ---- 处理 chunking 中的请求（续 chunk）----
    continuing_chunks = []
    remaining_chunks = []

    for chunked_req in self.chunking_reqs:
      req = chunked_req.req
      cached_len = chunked_req.cached_len + chunked_req.chunk_size
      remaining_len = req.total_input_len - cached_len

      # 更新 req 的 extend_input_len 供 add_chunked_req 使用
      req.extend_input_len = remaining_len
      still_truncated = adder.add_chunked_req(req)
      actual_chunk_size = req.extend_input_len

      new_chunked = ChunkedReq(req, cached_len, actual_chunk_size)
      continuing_chunks.append(new_chunked)
      if still_truncated is not None:
        # 还有后续 chunk，留到下一轮
        remaining_chunks.append(ChunkedReq(req, cached_len + actual_chunk_size, 0))
        logger.debug(
          f"Continuing chunked req_id={req.req_id}, "
          f"cached_len={cached_len}, "
          f"actual_chunk_size={actual_chunk_size}, "
          f"remaining_len={remaining_len - actual_chunk_size}"
        )
      else:
        logger.debug(
          f"Finished chunked req_id={req.req_id}, "
          f"cached_len={cached_len}, "
          f"actual_chunk_size={actual_chunk_size}"
        )

    self.chunking_reqs = remaining_chunks

    # ---- 从 waiting_queue 中添加新的 prefill 请求 ----
    prefill_reqs = []
    new_chunked_reqs = []
    remaining_waiting = []

    for req in self.waiting_queue:
      if adder.no_remaining_budget():
        remaining_waiting.append(req)
        continue

      # 计算 prefix 匹配
      self.kv_cache_mgr.prefix_for_waiting_req(req)

      result = adder.try_add_prefill(req)

      if result == AddReqResult.NO_TOKEN:
        # KV cache 不足，后续请求也无法添加
        remaining_waiting.append(req)
        remaining_waiting.extend(self.waiting_queue[self.waiting_queue.index(req) + 1 :])
        break
      elif result == AddReqResult.OTHER:
        # 软限制（chunk 预算用尽等），跳过但继续尝试
        remaining_waiting.append(req)
        continue

      # CONTINUE: 添加成功
      if adder.new_chunked_req is req:
        # 被截断为 chunk
        chunked_req = ChunkedReq(
          req=req,
          cached_len=req.cache_protected_len,
          chunk_size=req.extend_input_len,
        )
        new_chunked_reqs.append(chunked_req)
        adder.new_chunked_req = None  # reset
      else:
        prefill_reqs.append(req)

    self.waiting_queue = remaining_waiting

    # ---- 合并结果 ----
    all_extend_reqs = (
      prefill_reqs + [c.req for c in continuing_chunks] + [c.req for c in new_chunked_reqs]
    )
    all_chunked_reqs = continuing_chunks + new_chunked_reqs

    if not all_extend_reqs:
      return None

    # 更新 chunking_reqs（未完成的 chunk 留到下一轮）
    for chunked_req in all_chunked_reqs:
      if not chunked_req.is_last_chunk:
        self.chunking_reqs.append(chunked_req)

    # ---- 构建 mixed batch: prefill + decode ----
    # 当开启 chunked prefill 且有 running decode 请求时，
    # 将 decode 请求附加到 prefill 批次，组成 mixed batch，
    # 所有请求统一使用 extend kernel 处理（decode 视为 extend_len=1）
    decode_reqs = []
    if self.enable_chunked_prefill and running_reqs:
      decode_reqs = list(running_reqs)  # 复制列表，不修改原 running_batch

    # 合并所有请求: prefill extend + decode (as extend_len=1)
    all_batch_reqs = all_extend_reqs + decode_reqs

    batch = ScheduledBatch.init_new(all_batch_reqs, device=device)
    batch.decoding_reqs = decode_reqs if decode_reqs else None

    if decode_reqs:
      batch.forward_mode = ForwardMode.MIXED
    else:
      batch.forward_mode = ForwardMode.EXTEND

    # self._prepare_chunked_input_ids(batch, all_chunked_reqs)
    # --- overlap 路径：将 running_batch 的 placeholder metadata 透传给 prepare_for_mixed ---
    mixed_kwargs = {}
    if decode_reqs:
      # Always pass seq_lens metadata so KVCacheManager can update running_batch
      # lengths in-place (like prepare_for_decode) without any CUDA `.item()` fallbacks.
      if self.running_batch.seq_lens is None or self.running_batch.seq_lens_cpu is None:
        self._rebuild_running_batch_metadata()
      mixed_kwargs["decode_seq_lens"] = self.running_batch.seq_lens
      mixed_kwargs["decode_seq_lens_cpu"] = self.running_batch.seq_lens_cpu
      if skip_decode_input_ids and self.running_batch.output_ids is not None:
        mixed_kwargs["decode_input_ids"] = self.running_batch.output_ids
    self.kv_cache_mgr.prepare_for_mixed(batch, all_extend_reqs, decode_reqs, **mixed_kwargs)
    batch.uses_placeholder = bool("decode_input_ids" in mixed_kwargs)

    # 非 chunked 请求完成 prefill 后加入 running_batch
    for req in all_extend_reqs:
      if not req.is_chunked:
        self.running_batch.reqs.append(req)
        self._update_running_batch_metadata(batch, req)

    return batch

  # ======================== Decode 批次构建 ========================

  def _get_decode_batch(
    self, device: torch.device, *, skip_decode_input_ids: bool = False
  ) -> Optional[ScheduledBatch]:
    """
    构建 decode 批次

    核心防 OOM 逻辑:
    1. check_decode_mem: 估算下一步需要的 KV page 数
    2. 如果不足 → 收缩批次 (retract) 并调高 new_token_ratio
    3. 如果连续多轮没有 retract → 逐步降低 new_token_ratio
    4. alloc_for_decode: 按 page_size 追加 KV 内存
    """
    if len(self.running_batch.reqs) == 0:
      return None

    # ---- 内存检查 + 收缩 ----
    retracted = self._check_decode_mem()

    if retracted:
      # 有收缩 → 调高 new_token_ratio（给 decode 更多预留）
      self.num_continuous_retract += 1
      self.num_continuous_decode_ok = 0
      if self.num_continuous_retract >= _RETRACT_THRESHOLD:
        old_ratio = self.new_token_ratio
        self.new_token_ratio = min(self.new_token_ratio + _NTR_GROW_STEP, _MAX_NEW_TOKEN_RATIO)
        if self.new_token_ratio != old_ratio:
          logger.info(
            f"new_token_ratio raised: {old_ratio:.3f} → "
            f"{self.new_token_ratio:.3f} "
            f"(continuous_retract={self.num_continuous_retract})"
          )
    else:
      # 没有收缩 → 逐步降低 new_token_ratio（释放 prefill 空间）
      self.num_continuous_retract = 0
      self.num_continuous_decode_ok += 1
      if self.num_continuous_decode_ok >= 5:
        old_ratio = self.new_token_ratio
        self.new_token_ratio = max(self.new_token_ratio - _NTR_DECAY_STEP, _MIN_NEW_TOKEN_RATIO)
        self.num_continuous_decode_ok = 0
        if self.new_token_ratio != old_ratio:
          logger.debug(f"new_token_ratio lowered: {old_ratio:.3f} → {self.new_token_ratio:.3f}")

    if len(self.running_batch.reqs) == 0:
      return None

    # ---- 为 decode 分配 KV 内存 ----
    # 只有在 placeholder overlap 路径下，才需要跳过从 req.output_ids[-1] 构建 input_ids，
    # 以避免 schedule(N) 依赖 process(N-1)（输出 token 尚未写回 req.output_ids）。
    self.kv_cache_mgr.prepare_for_decode(
      self.running_batch, skip_input_ids=bool(skip_decode_input_ids)
    )
    self.running_batch.uses_placeholder = bool(skip_decode_input_ids)
    self.running_batch.forward_mode = ForwardMode.DECODE
    return self.running_batch

  def _check_decode_mem(self) -> bool:
    """
    检查 decode 内存是否充足，不足时收缩批次

    估算: 每个 decode 请求下一步需要 1 个 token，
    如果需要跨 page 边界则需要分配新的 page。

    Returns:
        True 如果有请求被收缩 (retracted)
    """
    if len(self.running_batch.reqs) == 0:
      return False

    available = self.kv_cache_mgr.available_tokens()
    bs = len(self.running_batch.reqs)

    # 精确估算 decode 需要的新 page 数
    # 当下一步 token 跨越 page 边界时需要分配整个新 page。
    # 条件等价变换：
    #   (seq_len_next % page_size == 1)  <=>  (seq_len % page_size == 0)
    # 直接用当前 seq_lens_cpu 避免额外的张量加法开销。
    if self.running_batch.seq_lens_cpu is not None:
      seq_lens_cpu = self.running_batch.seq_lens_cpu
      num_new_pages = (seq_lens_cpu.remainder(self.page_size) == 0).sum().item()
      needed = int(num_new_pages) * self.page_size
    else:
      # 回退到保守估算：最坏情况每个请求都需要新 page
      needed = bs * self.page_size

    logger.debug(f"Check decode mem: available={available}, needed={needed}, running_bs={bs}")

    if available >= needed:
      return False

    # 内存不足 → 从 batch 尾部开始收缩
    retract_count = 0
    while len(self.running_batch.reqs) > 0:
      # 重新估算剩余请求的实际需求
      remaining_bs = len(self.running_batch.reqs)
      if self.running_batch.seq_lens_cpu is not None and remaining_bs > 0:
        remaining_seq_lens = self.running_batch.seq_lens_cpu[:remaining_bs]
        remaining_pages = (remaining_seq_lens.remainder(self.page_size) == 0).sum().item()
        remaining_needed = int(remaining_pages) * self.page_size
      else:
        remaining_needed = remaining_bs * self.page_size

      if available >= remaining_needed:
        break

      req = self.running_batch.reqs.pop()
      retract_count += 1

      # BUG Fix：如果该请求在 pending_release_reqs 中（overlap 场景下 schedule 先于 process 执行），
      # 需要先移除，避免后续 drain 时出现 req_pool_idx=-1 的异常
      if req in self.pending_release_reqs:
        self.pending_release_reqs.remove(req)

      # 释放这个请求的 KV cache
      self.kv_cache_mgr.release_request(req, is_insert=True)

      # 标记为被收缩，重新放回 waiting 队列头部
      req.is_retracted = True
      req.ever_retracted = True
      req.finished = False
      req.is_chunked = False
      # 重置 KV 相关状态，下次重新 prefill
      req.output_ids = []
      req.fill_ids = []
      req.prefix_indices = None
      req.cache_protected_len = 0
      req.last_node = None
      req.extend_input_len = 0
      req.req_pool_idx = -1  # 重置 req_pool_idx，避免读取已释放的数据
      # 重置缓存的元数据
      req._cached_seq_len = len(req.origin_input_ids)
      req._cached_last_token = None
      self.waiting_queue.insert(0, req)

      # 更新可用量
      available = self.kv_cache_mgr.available_tokens()

    # 同步 running_batch 的元数据
    if retract_count > 0:
      self._rebuild_running_batch_metadata()
      self.num_retracted_reqs += retract_count
      logger.warning(
        f"Decode retract: removed {retract_count} reqs, "
        f"running_bs={len(self.running_batch.reqs)}, "
        f"available={available}"
      )

    return retract_count > 0

  def _rebuild_running_batch_metadata(self):
    """收缩后重建 running_batch 的元数据"""
    batch = self.running_batch
    if len(batch.reqs) == 0:
      batch.req_pool_indices = None
      batch.seq_lens = None
      batch.seq_lens_cpu = None
      batch.output_ids = None
      return

    # 从 req 上的 req_pool_idx 重建
    # 优化：使用预分配的 tensor 和批量赋值
    bs = len(batch.reqs)
    device = batch.req_pool_indices.device if batch.req_pool_indices is not None else "cuda"

    pool_indices = torch.empty(bs, dtype=torch.int64, device=device)
    seq_lens = torch.empty(bs, dtype=torch.int64, device=device)
    seq_lens_cpu = torch.empty(bs, dtype=torch.int64)

    for i, req in enumerate(batch.reqs):
      pool_indices[i] = req.req_pool_idx
      # 使用缓存的 current_seq_len 而非重复计算
      seq_lens[i] = req.current_seq_len
      seq_lens_cpu[i] = req.current_seq_len

    batch.req_pool_indices = pool_indices
    batch.seq_lens = seq_lens
    batch.seq_lens_cpu = seq_lens_cpu

    # retract 从尾部 pop，output_ids 需要同步截断（存储了 future placeholder）
    if batch.output_ids is not None:
      batch.output_ids = batch.output_ids[: len(batch.reqs)]

  # ======================== private helper function ========================
  def _update_running_batch_metadata(self, batch: ScheduledBatch, req: Req):
    """当请求从 extend batch 移到 running_batch 时同步元数据"""
    try:
      req_idx_in_batch = batch.reqs.index(req)
    except ValueError:
      return

    if batch.req_pool_indices is None or batch.seq_lens is None:
      return

    req_pool_idx = batch.req_pool_indices[req_idx_in_batch : req_idx_in_batch + 1]
    seq_len = batch.seq_lens[req_idx_in_batch : req_idx_in_batch + 1]
    seq_len_cpu = (
      batch.seq_lens_cpu[req_idx_in_batch : req_idx_in_batch + 1]
      if batch.seq_lens_cpu is not None
      else None
    )

    if self.running_batch.req_pool_indices is None:
      self.running_batch.req_pool_indices = req_pool_idx
      self.running_batch.seq_lens = seq_len
      self.running_batch.seq_lens_cpu = seq_len_cpu
    else:
      self.running_batch.req_pool_indices = torch.cat(
        [self.running_batch.req_pool_indices, req_pool_idx]
      )
      self.running_batch.seq_lens = torch.cat([self.running_batch.seq_lens, seq_len])
      if seq_len_cpu is not None and self.running_batch.seq_lens_cpu is not None:
        self.running_batch.seq_lens_cpu = torch.cat([self.running_batch.seq_lens_cpu, seq_len_cpu])

  def _filter_batch(self, batch: ScheduledBatch):
    """过滤掉已完成的请求，并同步更新相关元数据"""
    if not batch.reqs:
      return

    original_len = len(batch.reqs)

    keep_indices = []
    kept_reqs = []
    for i, req in enumerate(batch.reqs):
      if not req.finished:
        keep_indices.append(i)
        kept_reqs.append(req)

    batch.reqs = kept_reqs

    if len(keep_indices) == original_len:
      return

    if len(keep_indices) > 0 and batch.req_pool_indices is not None:
      # 优化：使用 torch.as_tensor 避免 CUDA 同步
      keep_indices_cpu = torch.as_tensor(keep_indices, dtype=torch.int64)
      keep_indices_tensor = keep_indices_cpu.to(batch.req_pool_indices.device, non_blocking=True)
      batch.req_pool_indices = batch.req_pool_indices[keep_indices_tensor]
      batch.seq_lens = batch.seq_lens[keep_indices_tensor]
      if batch.seq_lens_cpu is not None:
        batch.seq_lens_cpu = batch.seq_lens_cpu[keep_indices]
      # BUG Fix: overlap by future map, 同步过滤 output_ids（存储了 future indices 占位符）
      if batch.output_ids is not None:
        batch.output_ids = batch.output_ids[keep_indices_tensor]
    elif len(keep_indices) == 0:
      batch.req_pool_indices = None
      batch.seq_lens = None
      batch.seq_lens_cpu = None
      batch.output_ids = None

  # ======================== 前向 & 结果处理 ========================

  def run_batch(self, batch: ScheduledBatch) -> Any:
    forward_batch = ForwardBatch.init_new(batch)
    out = self.modelrunner.forward(forward_batch)
    logits_output = out.logits
    next_token_ids = self.modelrunner.sample(logits_output, forward_batch)

    return BatchResult(
      logits=logits_output,
      next_token_ids=next_token_ids,
    )

  def process_batch_result(self, batch: ScheduledBatch, result: BatchResult) -> List[str]:
    """处理批次推理结果"""
    if batch is None or len(batch.reqs) == 0:
      return []

    next_token_ids = result.next_token_ids
    if isinstance(next_token_ids, torch.Tensor):
      # Avoid redundant .cpu() if already on CPU (e.g., from overlap pinned buffer)
      if next_token_ids.device.type != "cpu":
        logger.error(
          "Expected next_token_ids to be on CPU, but got device: %s", next_token_ids.device
        )
        next_token_ids = next_token_ids.cpu()
      next_token_ids = next_token_ids.tolist()

    output_texts = []
    finished_req_ids = []
    for i, req in enumerate(batch.reqs):
      token_id = next_token_ids[i]
      req.append_output_token(token_id)

      delta_text, is_finished = self.incremental_decoder.decode(
        req_id=req.req_id,
        token_id=token_id,
        eos_token_id=self.eos_token_id,
      )
      output_texts.append(delta_text)

      if len(req.output_ids) >= req.max_tokens:
        is_finished = True
        remaining = self.incremental_decoder.flush(req.req_id)
        if remaining:
          output_texts[-1] += remaining

      if is_finished:
        req.finished = True
        if token_id == self.eos_token_id:
          req.finished_reason = "eos"
        else:
          req.finished_reason = "max_tokens"
        finished_req_ids.append(req.req_id)

    self._handle_finished_requests(batch, finished_req_ids)
    return output_texts

  def _handle_finished_requests(self, batch: ScheduledBatch, finished_req_ids: List[int]):
    """处理已完成的请求：释放 KV cache，移入 finished 列表"""
    if not finished_req_ids:
      return

    finished_req_id_set = set(finished_req_ids)
    for req in batch.reqs:
      if req.req_id in finished_req_id_set:
        # 延迟释放：将 radix cache 更新 + KV 释放移出主循环的 process_result，
        # 由上层（LLMEngine）在 GPU 计算期间批量 drain，以便实现 overlap。
        self.pending_release_reqs.append(req)
        self.finished_reqs.append(req)

  def drain_pending_releases(self, max_reqs: Optional[int] = None) -> int:
    """
    Drain deferred releases. This updates radix cache and frees KV pages.

    Intended to be called by the engine while GPU compute is in-flight.

    Args:
        max_reqs: Optional cap to release at most N requests.

    Returns:
        Number of requests released.
    """
    if not self.pending_release_reqs:
      return 0

    if max_reqs is None or max_reqs <= 0:
      to_release = self.pending_release_reqs
      self.pending_release_reqs = []
    else:
      to_release = self.pending_release_reqs[:max_reqs]
      self.pending_release_reqs = self.pending_release_reqs[max_reqs:]

    # Batch frees to reduce allocator bookkeeping overhead.
    allocator = getattr(self.kv_cache_mgr, "token_allocator", None)
    begin_group = getattr(allocator, "begin_free_group", None)
    end_group = getattr(allocator, "end_free_group", None)
    if begin_group is not None and end_group is not None:
      begin_group()
      try:
        for req in to_release:
          # 防御性检查：overlap 场景下，schedule(N) 中的 retract 可能已释放并重置了
          # req_pool_idx，此时跳过，避免重复释放或无效访问
          # if int(getattr(req, "req_pool_idx", -1)) < 0:
          #   logger.debug(
          #     f"drain_pending_releases: skip req_id={req.req_id} with "
          #     f"req_pool_idx={getattr(req, 'req_pool_idx', -1)} (already released)"
          #   )
          #   continue
          assert req.req_pool_idx != -1, (
            f"Pending release req_id={req.req_id} has invalid req_pool_idx={getattr(req, 'req_pool_idx', -1)}"
          )
          self.kv_cache_mgr.update_finished_req_radix_cache(req)
          req.req_pool_idx = -1
      finally:
        end_group()
    else:
      for req in to_release:
        # # 防御性检查：overlap 场景下，schedule(N) 中的 retract 可能已释放并重置了
        # # req_pool_idx，此时跳过，避免重复释放或无效访问
        # if int(getattr(req, "req_pool_idx", -1)) < 0:
        #   logger.debug(
        #     f"drain_pending_releases: skip req_id={req.req_id} with "
        #     f"req_pool_idx={getattr(req, 'req_pool_idx', -1)} (already released)"
        #   )
        #   continue
        assert req.req_pool_idx != -1, (
          f"Pending release req_id={req.req_id} has invalid req_pool_idx={getattr(req, 'req_pool_idx', -1)}"
        )
        self.kv_cache_mgr.update_finished_req_radix_cache(req)
        req.req_pool_idx = -1

    return len(to_release)

  # ======================== 调试和监控 ========================

  def _print_scheduler_info(self):
    """打印当前调度器状态，包括空闲页数"""
    free_pages = len(self.kv_cache_mgr.token_allocator.free_pages)
    total_pages = self.kv_cache_mgr.token_allocator.num_pages
    used_pages = total_pages - free_pages
    available_tokens = self.kv_cache_mgr.available_tokens()

    logger.debug(
      f"\n{'=' * 80}\n"
      f"[Scheduler Round] KV Cache Status:\n"
      f"  Free Pages: {free_pages}/{total_pages} "
      f"(Used: {used_pages}, {used_pages / total_pages * 100:.1f}%)\n"
      f"  Available Tokens: {available_tokens}\n"
      f"  Page Size: {self.page_size}\n"
      f"  Waiting Queue: {len(self.waiting_queue)} reqs\n"
      f"  Running Batch: {len(self.running_batch.reqs)} reqs\n"
      f"  Chunking Reqs: {len(self.chunking_reqs)} reqs\n"
      f"  New Token Ratio: {self.new_token_ratio:.3f}\n"
      f"{'=' * 80}"
    )

  def _print_batch_info(self, batch: ScheduledBatch, batch_type: str):
    """打印要执行的 batch 详细信息"""
    if batch is None:
      return

    logger.debug(f"\n[Batch Info] Type: {batch_type}")
    logger.debug(f"  Forward Mode: {batch.forward_mode.name}")
    logger.debug(f"  Batch Size: {len(batch.reqs)}")

    if batch.input_ids is not None:
      logger.debug(f"  Input IDs shape: {batch.input_ids.shape}")

    # 打印每个请求的详细信息
    logger.debug("  Requests Details:")
    for i, req in enumerate(batch.reqs):
      input_len = len(req.origin_input_ids)
      output_len = len(req.output_ids)
      total_len = input_len + output_len

      req_info = (
        f"    [{i}] req_id={req.req_id}, "
        f"input_len={input_len}, output_len={output_len}, "
        f"total_len={total_len}"
      )

      # 如果是 chunked 请求，添加额外信息
      if req.is_chunked:
        req_info += f", chunked_prefill_len={req.chunked_prefill_len}"

      # 如果有 extend_input_len，打印
      if hasattr(req, "extend_input_len") and req.extend_input_len > 0:
        req_info += f", extend_len={req.extend_input_len}"

      logger.debug(req_info)

    # 如果是 MIXED batch，额外打印 decode 请求信息
    if batch.decoding_reqs:
      logger.info(f"  Decoding Requests: {len(batch.decoding_reqs)}")

    # 打印 seq_lens 如果有的话
    if batch.seq_lens is not None:
      logger.info(
        f"  Seq Lens: {batch.seq_lens.tolist() if batch.seq_lens.numel() <= 10 else f'[{batch.seq_lens.numel()} items]'}"
      )

    logger.info(f"{'-' * 80}")

  # ======================== 监控 API ========================

  def get_budget_stats(self) -> dict:
    """获取当前 token 预算统计信息"""
    running_bs = len(self.running_batch.reqs)
    available = self.kv_cache_mgr.available_tokens()
    return {
      "max_extend_len": self.max_extend_len,
      "running_bs": running_bs,
      "waiting_queue_len": len(self.waiting_queue),
      "chunking_reqs": len(self.chunking_reqs),
      "available_kv_tokens": available,
      "new_token_ratio": self.new_token_ratio,
      "num_continuous_retract": self.num_continuous_retract,
      "num_retracted_total": self.num_retracted_reqs,
      "decode_waiting_ticks": self.decode_waiting_ticks,
      "enable_chunked_prefill": self.enable_chunked_prefill,
    }

  def get_request_output(self, req_id: int) -> Optional[str]:
    """获取指定请求的完整输出文本"""
    return self.incremental_decoder.get_full_text(req_id)
