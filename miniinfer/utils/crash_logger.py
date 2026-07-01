"""推理崩溃时的上下文 dump 工具。

当一次推理 step 抛出异常时，单凭 traceback 往往不够定位问题——需要同时
看到调度器状态（running/waiting 队列、KV cache 占用）、当前 ScheduledBatch
/ ForwardBatch 的元数据、以及 attention backend 的 forward_metadata（block
table、cache_seqlens 等）。本模块把这些信息收集到一条结构化日志里。

所有字段都用 ``getattr`` 防御性读取：dump 本身绝不能再抛异常把原始错误
吞掉。tensor 一律只取 ``shape/dtype/device`` 与少量统计量，避免把整张
block table 打成日志。
"""
from __future__ import annotations

from typing import Any, Optional

from miniinfer.utils import get_logger

logger = get_logger("miniinfer")


def _tinfo(t: Any) -> str:
  """tensor 的安全摘要：shape/dtype/device + 数值统计。"""
  if t is None:
    return "None"
  try:
    import torch

    if isinstance(t, torch.Tensor):
      s = f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
      if t.numel() > 0 and t.is_floating_point():
        s += f" min={t.float().min().item():.4g} max={t.float().max().item():.4g} mean={t.float().mean().item():.4g}"
      elif t.numel() > 0 and t.dtype in (torch.int32, torch.int64, torch.int16, torch.int8):
        s += f" min={t.min().item()} max={t.max().item()}"
      return s
  except Exception:
    pass
  return repr(t)


def _dump_scheduler(scheduler: Any) -> None:
  if scheduler is None:
    return
  logger.error("==== Scheduler state ====")
  try:
    waiting = getattr(scheduler, "waiting_queue", None)
    running = getattr(scheduler, "running_batch", None)
    pending = getattr(scheduler, "pending_release_reqs", None)
    running_reqs = getattr(running, "reqs", None)
    logger.error(
      f"waiting_queue={len(waiting) if waiting is not None else 'N/A'} "
      f"running_batch.reqs={len(running_reqs) if running_reqs is not None else 'N/A'} "
      f"pending_release={len(pending) if pending is not None else 'N/A'}"
    )
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to summarize scheduler queues: {e!r})")

  # 逐个 running request 的关键状态
  try:
    running = getattr(scheduler, "running_batch", None)
    reqs = getattr(running, "reqs", None) or []
    for r in reqs[:16]:
      rid = getattr(r, "req_id", "?")
      prompt_idx = getattr(r, "prompt_idx", "?")
      origin_len = getattr(r, "origin_input_ids_len", None)
      cur_len = getattr(r, "output_ids", None)
      cur_len = len(cur_len) if hasattr(cur_len, "__len__") else "?"
      logger.error(
        f"  running req_id={rid} prompt_idx={prompt_idx} "
        f"prompt_len={origin_len} generated={cur_len}"
      )
    if len(reqs) > 16:
      logger.error(f"  ... ({len(reqs) - 16} more running reqs)")
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to enumerate running reqs: {e!r})")

  # KV cache 占用
  _dump_kv_cache(getattr(scheduler, "kv_cache_mgr", None))


def _dump_kv_cache(kv_cache_mgr: Any) -> None:
  if kv_cache_mgr is None:
    return
  try:
    page_size = getattr(kv_cache_mgr, "page_size", "?")
    device = getattr(kv_cache_mgr, "device", "?")
    alloc = getattr(kv_cache_mgr, "token_allocator", None)
    free_pages = getattr(alloc, "free_pages", None)
    num_pages = getattr(kv_cache_mgr, "num_pages", None)
    if num_pages is None and free_pages is not None:
      # 退而求其次：只报告剩余页
      num_pages = "?"
    free_cnt = "?"
    total_cnt = "?"
    try:
      avail = alloc.available_size() if alloc is not None and hasattr(alloc, "available_size") else None
      free_cnt = int(free_pages.numel()) if free_pages is not None and hasattr(free_pages, "numel") else "?"
    except Exception:
      avail = None
    logger.error(
      f"KVCache: device={device} page_size={page_size} num_pages={num_pages} "
      f"free_pages={free_cnt} allocator.available_size={avail}"
    )
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to summarize kv cache: {e!r})")


def _dump_scheduled_batch(batch: Any) -> None:
  if batch is None:
    return
  logger.error("==== ScheduledBatch ====")
  try:
    logger.error(
      f"forward_mode={getattr(batch, 'forward_mode', '?')} "
      f"num_reqs={len(getattr(batch, 'reqs', []) or [])}"
    )
    for attr in ("seq_lens", "prefix_lens", "extend_lens", "extend_seq_lens"):
      v = getattr(batch, attr, None)
      if v is not None:
        logger.error(f"  {attr}: {_tinfo(v)}")
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to summarize scheduled batch: {e!r})")


def _dump_forward_batch(fb: Any) -> None:
  if fb is None:
    return
  logger.error("==== ForwardBatch ====")
  try:
    logger.error(
      f"forward_mode={getattr(fb, 'forward_mode', '?')} "
      f"batch_size={getattr(fb, 'batch_size', '?')}"
    )
    for attr in (
      "input_ids", "positions", "req_pool_indices", "out_cache_loc",
      "seq_lens", "extend_seq_lens", "extend_prefix_lens",
      "extend_seq_lens_cpu", "sampling_temperatures", "sampling_top_ps",
      "sampling_top_ks", "last_token_indices",
    ):
      v = getattr(fb, attr, None)
      if v is not None:
        logger.error(f"  {attr}: {_tinfo(v)}")
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to summarize forward batch: {e!r})")

  # attention backend 的 forward_metadata
  backend = getattr(fb, "attn_backend", None)
  _dump_attn_backend(backend)


def _dump_attn_backend(backend: Any) -> None:
  if backend is None:
    return
  logger.error("==== Attention backend ====")
  try:
    logger.error(f"backend={backend.type() if hasattr(backend, 'type') else type(backend).__name__}")
    logger.error(f"page_size={getattr(backend, 'page_size', '?')}")
    md = getattr(backend, "forward_metadata", None)
    if md is None:
      logger.error("forward_metadata=None (init_forward_metadata 未运行或已清理)")
      return
    logger.error(f"forward_metadata type={type(md).__name__}")
    for attr in (
      "block_table", "page_table", "cache_seqlens_int32", "cache_seqlens",
      "max_seq_len_k", "max_seq_len_q", "cu_seqlens_q", "cu_seqlens_k",
      "slot_mapping", "seq_lens",
    ):
      v = getattr(md, attr, None)
      if v is not None:
        logger.error(f"  metadata.{attr}: {_tinfo(v)}")
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to summarize attention backend: {e!r})")


def _dump_model_runner(model_runner: Any) -> None:
  if model_runner is None:
    return
  logger.error("==== ModelRunner ====")
  try:
    model = getattr(model_runner, "model", None)
    model_name = type(model).__name__ if model is not None else "None"
    cfg = getattr(model_runner, "model_config", None)
    model_type = getattr(cfg, "model_type", "?")
    logger.error(
      f"model={model_name} model_type={model_type} "
      f"device={getattr(model_runner, 'device', '?')} "
      f"use_cuda_graph={getattr(model_runner, 'use_cuda_graph', '?')}"
    )
    if cfg is not None:
      for attr in (
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size",
      ):
        v = getattr(cfg, attr, None)
        if v is not None:
          logger.error(f"  config.{attr}={v}")
      # MoE 相关
      num_experts = getattr(cfg, "num_experts", None)
      if num_experts:
        logger.error(
          f"  config.num_experts={num_experts} "
          f"num_experts_per_tok={getattr(cfg, 'num_experts_per_tok', '?')} "
          f"moe_intermediate_size={getattr(cfg, 'moe_intermediate_size', '?')}"
        )
    cg = getattr(model_runner, "cuda_graph_runner", None)
    if cg is not None:
      logger.error(
        f"  cuda_graph_runner={type(cg).__name__} "
        f"available={cg.is_available() if hasattr(cg, 'is_available') else '?'}"
      )
  except Exception as e:  # noqa: BLE001
    logger.error(f"  (failed to summarize model runner: {e!r})")


def log_inference_crash(
  exc: BaseException,
  *,
  scheduler: Any = None,
  model_runner: Any = None,
  forward_batch: Any = None,
  scheduled_batch: Any = None,
  stage: Optional[str] = None,
) -> None:
  """在推理 step 崩溃时打印尽可能详细的上下文，然后由调用方 re-raise。

  不会吞掉异常，只负责打印。``logger.exception`` 会附带完整 traceback。
  """
  where = f" [{stage}]" if stage else ""
  logger.error(f"========== Inference step crashed{where} ==========")
  # 先打 traceback（logger.exception 用 ERROR 级别 + exc_info）
  try:
    logger.error(f"Exception: {type(exc).__name__}: {exc}", exc_info=exc)
  except Exception:
    logger.error(f"Exception: {type(exc).__name__}: {exc}")

  _dump_model_runner(model_runner)
  _dump_scheduler(scheduler)
  _dump_scheduled_batch(scheduled_batch)
  _dump_forward_batch(forward_batch)
  logger.error("========== End of crash context ==========")
