"""
Multi-Batch KV Cache Consistency Test

Compare outputs when running multiple requests:
A) Each request run independently (batch_size=1)
B) Multiple requests run together in same batch (batch_size>1)

If KV cache management has issues with multi-batch, the outputs will differ.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo paths
ROOT = Path(__file__).resolve().parents[2]
MINIINFER_ROOT = ROOT / "miniinfer"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(MINIINFER_ROOT) not in sys.path:
  sys.path.insert(0, str(MINIINFER_ROOT))

from config.engine.config import EngineConfig
from engine.model_runner import ModelRunner
from scheduler.scheduler_batch import Req, ScheduledBatch, ForwardBatch
from kvcache.kv_cache_manager import KVCacheManager
from layers.attention_backend.flashattention_backend import FlashAttention2Backend
from utils.sampling_params import SamplingParams


def _build_kv_cache_manager(
  cfg: EngineConfig,
  size: int,
  max_context_len: int,
  max_requests: int,
  page_size: int,
  device: str,
) -> KVCacheManager:
  return KVCacheManager(
    size=size,
    max_requests=max_requests,
    max_context_len=max_context_len,
    num_layers=cfg.hf_config.num_hidden_layers,
    num_heads=cfg.hf_config.num_key_value_heads,
    head_dim=cfg.hf_config.hidden_size // cfg.hf_config.num_attention_heads,
    device=device,
    enable_prefix_cache=False,
    page_size=page_size,
  )


def _is_instruct_like_model(model_name: str) -> bool:
  name = model_name.lower()
  return ("instruct" in name) or ("chat" in name)


def _should_apply_chat_template(tokenizer, model_name: str) -> bool:
  if not hasattr(tokenizer, "apply_chat_template"):
    return False
  if not getattr(tokenizer, "chat_template", None):
    return False
  return _is_instruct_like_model(model_name)


def _encode_prompt(tokenizer, model_name: str, prompt: str) -> List[int]:
  if _should_apply_chat_template(tokenizer, model_name):
    messages = [{"role": "user", "content": prompt}]
    try:
      encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
      )
      if isinstance(encoded, torch.Tensor):
        return encoded.tolist()
      if isinstance(encoded, dict) and "input_ids" in encoded:
        ids = encoded["input_ids"]
        if isinstance(ids, torch.Tensor):
          return ids.tolist()
        return list(ids)
      return list(encoded)
    except Exception as exc:
      print(f"[WARN] apply_chat_template failed ({exc}); fallback to tokenizer.encode")
  return tokenizer.encode(prompt)


def _try_add_token_id(
  tokenizer,
  token: str,
  out: Set[int],
):
  try:
    token_id = tokenizer.convert_tokens_to_ids(token)
  except Exception:
    return
  if token_id is None:
    return
  try:
    resolved = tokenizer.convert_ids_to_tokens(int(token_id))
  except Exception:
    resolved = None
  if resolved == token:
    out.add(int(token_id))


def _resolve_stop_token_ids(tokenizer) -> Set[int]:
  stop_ids: Set[int] = set()
  if getattr(tokenizer, "eos_token_id", None) is not None:
    stop_ids.add(int(tokenizer.eos_token_id))

  _try_add_token_id(tokenizer, "<|im_end|>", stop_ids)
  _try_add_token_id(tokenizer, "<|endoftext|>", stop_ids)
  return stop_ids


def _required_kv_size_tokens(
  prompt_lens: List[int],
  decode_steps: int,
  page_size: int,
  safety_pages: int = 0,
) -> int:
  total_pages = 0
  for prompt_len in prompt_lens:
    final_len = max(1, int(prompt_len) + int(decode_steps))
    total_pages += max(1, math.ceil(final_len / page_size))
  total_pages += max(0, int(safety_pages))
  return total_pages * page_size


def _filter_finished_from_batch(batch: ScheduledBatch) -> None:
  if not batch.reqs:
    return

  keep_indices = [i for i, req in enumerate(batch.reqs) if not req.finished]
  if len(keep_indices) == len(batch.reqs):
    return

  batch.reqs = [batch.reqs[i] for i in keep_indices]

  if len(batch.reqs) == 0:
    batch.req_pool_indices = None
    batch.seq_lens = None
    batch.seq_lens_cpu = None
    return

  if batch.req_pool_indices is not None:
    idx = torch.tensor(keep_indices, dtype=torch.int64, device=batch.req_pool_indices.device)
    batch.req_pool_indices = batch.req_pool_indices[idx]
  if batch.seq_lens is not None:
    idx = torch.tensor(keep_indices, dtype=torch.int64, device=batch.seq_lens.device)
    batch.seq_lens = batch.seq_lens[idx]
  if batch.seq_lens_cpu is not None:
    batch.seq_lens_cpu = batch.seq_lens_cpu[keep_indices]


def _fmt_token(tokenizer, token_id: int) -> str:
  try:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
  except Exception:
    text = ""
  return f"{int(token_id)}:{text!r}"


def _check_kv_cache_layout(
  kv_mgr: KVCacheManager,
  batch: ScheduledBatch,
  page_size: int,
  label: str,
) -> Dict[int, List[int]]:
  """Check KV cache layout for each request and return mapping."""
  result = {}
  req_indices = batch.req_pool_indices.detach().cpu().tolist()
  seq_lens = batch.seq_lens.detach().cpu().tolist()

  print(f"\n[{label}] KV Cache Layout Check:")
  print(f"  req_pool_indices: {req_indices}")
  print(f"  seq_lens: {seq_lens}")
  print(
    f"  out_cache_loc: {batch.out_cache_loc.detach().cpu().tolist() if batch.out_cache_loc is not None else None}"
  )

  for i, req_idx in enumerate(req_indices):
    seq_len = int(seq_lens[i])
    kv_locs = kv_mgr.request_pool.read(req_idx, slice(0, seq_len)).detach().cpu().tolist()
    result[i] = kv_locs

    # Check for issues
    zeros = [j for j, loc in enumerate(kv_locs) if loc == 0]
    if zeros:
      print(f"  [WARN] req {i} (pool_idx={req_idx}): zero entries at positions {zeros}")

    # Check continuity within pages
    issues = []
    for j in range(1, len(kv_locs)):
      prev, cur = kv_locs[j - 1], kv_locs[j]
      if prev == 0 or cur == 0:
        continue
      prev_page = prev // page_size
      cur_page = cur // page_size
      if prev_page == cur_page:
        if cur != prev + 1:
          issues.append(f"non-contiguous at pos {j}: {prev}->{cur}")
      elif cur_page != prev_page + 1:
        issues.append(f"page jump at pos {j}: page {prev_page}->{cur_page}")

    if issues:
      print(f"  [WARN] req {i} (pool_idx={req_idx}): {issues}")
    else:
      print(
        f"  [OK] req {i} (pool_idx={req_idx}): seq_len={seq_len}, kv_locs={kv_locs[:5]}...{kv_locs[-3:] if len(kv_locs) > 5 else ''}"
      )

  return result


def _compare_kv_buffers(
  kv_mgr_single: KVCacheManager,
  kv_mgr_batch: KVCacheManager,
  locs_single: List[int],
  locs_batch: List[int],
  num_layers: int,
  rtol: float = 1e-3,
  atol: float = 1e-4,
) -> Tuple[bool, str]:
  """Compare KV buffer content between single and batch runs."""
  all_match = True
  messages = []

  for layer_id in range(num_layers):
    k_single, v_single = kv_mgr_single.get_kv_buffer(layer_id)
    k_batch, v_batch = kv_mgr_batch.get_kv_buffer(layer_id)

    for pos, (loc_s, loc_b) in enumerate(zip(locs_single, locs_batch)):
      k_s = k_single[loc_s].float()
      k_b = k_batch[loc_b].float()
      v_s = v_single[loc_s].float()
      v_b = v_batch[loc_b].float()

      k_match = torch.allclose(k_s, k_b, rtol=rtol, atol=atol)
      v_match = torch.allclose(v_s, v_b, rtol=rtol, atol=atol)

      if not (k_match and v_match):
        all_match = False
        k_diff = (k_s - k_b).abs()
        v_diff = (v_s - v_b).abs()
        messages.append(
          f"Layer {layer_id} pos {pos}: "
          f"K diff max/mean={k_diff.max().item():.4e}/{k_diff.mean().item():.4e}, "
          f"V diff max/mean={v_diff.max().item():.4e}/{v_diff.mean().item():.4e}"
        )
        if len(messages) >= 5:
          return all_match, "\n".join(messages)

  return all_match, "\n".join(messages) if messages else "All match"


def _reset_kv_manager_state(kv_mgr: KVCacheManager) -> None:
  """Reset allocator/request-pool state for a fresh single-request prefill run."""
  if kv_mgr.prefix_cache is not None:
    raise RuntimeError(
      "Recompute check requires enable_prefix_cache=False for deterministic reset."
    )
  kv_mgr.token_allocator.clear()
  kv_mgr.request_pool.req_to_token.zero_()
  kv_mgr.request_pool.free_slots = list(range(kv_mgr.request_pool.max_requests))


def _compute_next_logits_full_recompute(
  model_runner: ModelRunner,
  kv_mgr: KVCacheManager,
  attn_backend,
  input_ids: List[int],
  device: str,
  debug: bool = False,
) -> torch.Tensor:
  """Run one full prefill on input_ids and return logits of the last position."""
  req = Req(input_ids, SamplingParams(temperature=0, max_tokens=1))
  kv_mgr.prefix_for_waiting_req(req)
  batch = ScheduledBatch.init_new([req], device=device)
  kv_mgr.prepare_for_extend(batch)

  forward_batch = ForwardBatch.init_new(batch, attn_backend)
  attn_backend.init_forward_metadata(forward_batch)
  if debug:
    print(f"[DEBUG] input_ids (len={len(input_ids)}): {input_ids[:10]}...{input_ids[-5:]}")
    print(f"[DEBUG] forward_batch.input_ids: {forward_batch.input_ids.tolist()}")
    print(f"[DEBUG] forward_batch.positions: {forward_batch.positions.tolist()}")
    print(f"[DEBUG] batch.extend_lens: {batch.extend_lens}")
    print(f"[DEBUG] batch.prefix_lens: {batch.prefix_lens}")
    print(f"[DEBUG] req.fill_ids len: {len(req.fill_ids)}")
  out = model_runner.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
  )
  return out.logits[-1].detach().clone()


def _check_kv_decode_vs_full_recompute(
  model_runner: ModelRunner,
  kv_mgr: KVCacheManager,
  attn_backend,
  tokenizer,
  prompt_ids: List[int],
  kv_logits_list: List[torch.Tensor],
  kv_tokens: List[int],
  device: str,
  rtol: float = 1e-2,
  atol: float = 1e-2,
) -> Tuple[bool, List[str], List[int]]:
  """
  Compare KV decode against full recompute step-by-step.

  At step t, recompute logits from scratch on:
    prompt_ids + kv_tokens[:t]
  and compare with kv_logits_list[t].
  """
  details: List[str] = []
  recompute_tokens: List[int] = []
  ok = True

  generated_prefix: List[int] = []
  num_steps = min(len(kv_logits_list), len(kv_tokens))
  for step_idx in range(num_steps):
    _reset_kv_manager_state(kv_mgr)
    recompute_logits = _compute_next_logits_full_recompute(
      model_runner=model_runner,
      kv_mgr=kv_mgr,
      attn_backend=attn_backend,
      input_ids=prompt_ids + generated_prefix,
      device=device,
    ).float()

    kv_logits = kv_logits_list[step_idx].float()
    if kv_logits.shape != recompute_logits.shape:
      ok = False
      details.append(
        f"step {step_idx}: shape mismatch kv={tuple(kv_logits.shape)} vs recompute={tuple(recompute_logits.shape)}"
      )
      generated_prefix.append(int(kv_tokens[step_idx]))
      continue

    diff = (kv_logits - recompute_logits).abs()
    logits_close = torch.allclose(kv_logits, recompute_logits, rtol=rtol, atol=atol)

    kv_token = int(kv_tokens[step_idx])
    recompute_token = int(torch.argmax(recompute_logits).item())
    recompute_tokens.append(recompute_token)
    token_match = kv_token == recompute_token

    if not (logits_close and token_match):
      ok = False
      _, top5_kv = torch.topk(kv_logits, 5)
      _, top5_rc = torch.topk(recompute_logits, 5)
      details.append(
        f"step {step_idx}: token_match={token_match}, logits_close={logits_close}, "
        f"max_diff={diff.max().item():.4e}, mean_diff={diff.mean().item():.4e}, "
        f"kv={_fmt_token(tokenizer, kv_token)}, recompute={_fmt_token(tokenizer, recompute_token)}, "
        f"top5_kv={[_fmt_token(tokenizer, t.item()) for t in top5_kv]}, "
        f"top5_recompute={[_fmt_token(tokenizer, t.item()) for t in top5_rc]}"
      )

    generated_prefix.append(kv_token)

  return ok, details, recompute_tokens


def _hf_next_logits(
  hf_model,
  input_ids: List[int],
  device: str,
) -> torch.Tensor:
  input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
  attention_mask = torch.ones_like(input_tensor, dtype=torch.long, device=device)
  with torch.no_grad():
    logits = hf_model(input_ids=input_tensor, attention_mask=attention_mask).logits[0, -1]
  return logits.detach().clone()


def _hf_greedy_tokens_and_logits(
  hf_model,
  prompt_ids: List[int],
  steps: int,
  device: str,
  stop_token_ids: Optional[Set[int]] = None,
) -> Tuple[List[int], List[torch.Tensor]]:
  generated: List[int] = []
  logits_list: List[torch.Tensor] = []
  prefix = list(prompt_ids)

  for _ in range(steps):
    logits = _hf_next_logits(hf_model, prefix, device).float()
    token = int(torch.argmax(logits).item())
    logits_list.append(logits)
    generated.append(token)
    prefix.append(token)
    if stop_token_ids and token in stop_token_ids:
      break

  return generated, logits_list


def _teacher_forcing_align_miniinfer_vs_hf(
  model_runner: ModelRunner,
  kv_mgr: KVCacheManager,
  attn_backend,
  hf_model,
  tokenizer,
  prompt_ids: List[int],
  teacher_tokens: List[int],
  device: str,
  rtol: float = 1e-2,
  atol: float = 1e-2,
  topk: int = 5,
) -> Tuple[bool, List[str]]:
  """
  Teacher-forcing alignment:
  At step t, use the same prefix (prompt + teacher_tokens[:t]) and compare
  MiniInfer vs HF next-token logits.
  """
  details: List[str] = []
  ok = True
  prefix: List[int] = list(prompt_ids)

  for step_idx, teacher_token in enumerate(teacher_tokens):
    _reset_kv_manager_state(kv_mgr)
    # Debug on first step to see what's happening
    debug_this_step = step_idx == 0
    mi_logits = _compute_next_logits_full_recompute(
      model_runner=model_runner,
      kv_mgr=kv_mgr,
      attn_backend=attn_backend,
      input_ids=prefix,
      device=device,
      debug=debug_this_step,
    ).float()
    hf_logits = _hf_next_logits(hf_model, prefix, device).float()

    if mi_logits.shape != hf_logits.shape:
      ok = False
      details.append(
        f"step {step_idx}: shape mismatch miniinfer={tuple(mi_logits.shape)} vs hf={tuple(hf_logits.shape)}"
      )
      prefix.append(int(teacher_token))
      continue

    diff = (mi_logits - hf_logits).abs()
    logits_close = torch.allclose(mi_logits, hf_logits, rtol=rtol, atol=atol)
    mi_top1 = int(torch.argmax(mi_logits).item())
    hf_top1 = int(torch.argmax(hf_logits).item())
    top1_match = mi_top1 == hf_top1

    if not (logits_close and top1_match):
      ok = False
      k = min(topk, int(mi_logits.shape[-1]))
      _, topk_mi = torch.topk(mi_logits, k)
      _, topk_hf = torch.topk(hf_logits, k)
      details.append(
        f"step {step_idx}: top1_match={top1_match}, logits_close={logits_close}, "
        f"max_diff={diff.max().item():.4e}, mean_diff={diff.mean().item():.4e}, "
        f"teacher={_fmt_token(tokenizer, int(teacher_token))}, "
        f"miniinfer_top1={_fmt_token(tokenizer, mi_top1)}, "
        f"hf_top1={_fmt_token(tokenizer, hf_top1)}, "
        f"top{int(k)}_miniinfer={[_fmt_token(tokenizer, t.item()) for t in topk_mi]}, "
        f"top{int(k)}_hf={[_fmt_token(tokenizer, t.item()) for t in topk_hf]}"
      )

    prefix.append(int(teacher_token))

  return ok, details


def run_single_request(
  model_runner: ModelRunner,
  kv_mgr: KVCacheManager,
  attn_backend,
  prompt_ids: List[int],
  steps: int,
  device: str,
  page_size: int,
  stop_token_ids: Optional[Set[int]] = None,
  verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[int], List[int]]:
  """Run a single request and return logits, generated tokens, and KV locations."""
  req = Req(prompt_ids, SamplingParams(temperature=0, max_tokens=steps + 1))
  kv_mgr.prefix_for_waiting_req(req)

  batch = ScheduledBatch.init_new([req], device=device)
  kv_mgr.prepare_for_extend(batch)

  forward_batch = ForwardBatch.init_new(batch, attn_backend)
  attn_backend.init_forward_metadata(forward_batch)

  # Debug output for prefill
  if verbose:
    print(f"[SINGLE PREFILL] prompt_ids len={len(prompt_ids)}")
    print(f"  forward_batch.input_ids: {forward_batch.input_ids.tolist()}")
    print(f"  forward_batch.positions: {forward_batch.positions.tolist()}")
    print(f"  batch.extend_lens: {batch.extend_lens}")
    print(f"  batch.prefix_lens: {batch.prefix_lens}")
    print(f"  req.fill_ids len: {len(req.fill_ids)}")

  # Prefill
  out = model_runner.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
  )

  logits_list = [out.logits[-1].detach().clone()]
  next_token = int(torch.argmax(out.logits[-1]).item())
  req.output_ids.append(next_token)
  generated_tokens = [next_token]
  if stop_token_ids and next_token in stop_token_ids:
    req.finished = True

  # Decode steps
  for step in range(1, steps):
    if req.finished:
      break
    kv_mgr.prepare_for_decode(batch)
    forward_batch = ForwardBatch.init_new(batch, attn_backend)
    attn_backend.init_forward_metadata(forward_batch)

    out = model_runner.model.forward(
      forward_batch.input_ids,
      forward_batch.positions,
      forward_batch,
    )

    logits_i = out.logits[0].detach().clone()
    logits_list.append(logits_i)
    next_token = int(torch.argmax(logits_i).item())
    req.output_ids.append(next_token)
    generated_tokens.append(next_token)
    if stop_token_ids and next_token in stop_token_ids:
      req.finished = True

    if verbose and step <= 5:
      _, top5 = torch.topk(logits_i, 5)
      print(
        f"  [SINGLE] Step {step}: next_token={next_token}, top5={top5.tolist()}, logits_mean={logits_i.mean().item():.4f}, logits_std={logits_i.std().item():.4f}"
      )

  # Get final KV locations
  seq_len = len(prompt_ids) + len(req.output_ids)
  kv_locs = kv_mgr.request_pool.read(req.req_pool_idx, slice(0, seq_len)).detach().cpu().tolist()

  if verbose:
    _check_kv_cache_layout(kv_mgr, batch, page_size, "single")

  return logits_list, generated_tokens, kv_locs


def run_batch_requests(
  model_runner: ModelRunner,
  kv_mgr: KVCacheManager,
  attn_backend,
  all_prompt_ids: List[List[int]],
  steps: int,
  device: str,
  page_size: int,
  stop_token_ids: Optional[Set[int]] = None,
  verbose: bool = False,
) -> List[Tuple[List[torch.Tensor], List[int], List[int]]]:
  """Run multiple requests in a batch and return per-request results."""
  batch_size = len(all_prompt_ids)

  # Create requests
  reqs = [
    Req(prompt_ids, SamplingParams(temperature=0, max_tokens=steps + 1))
    for prompt_ids in all_prompt_ids
  ]
  req_to_idx = {id(req): i for i, req in enumerate(reqs)}

  for req in reqs:
    kv_mgr.prefix_for_waiting_req(req)

  batch = ScheduledBatch.init_new(reqs, device=device)
  kv_mgr.prepare_for_extend(batch)

  forward_batch = ForwardBatch.init_new(batch, attn_backend)
  attn_backend.init_forward_metadata(forward_batch)

  if verbose:
    print(f"\n[BATCH PREFILL] batch_size={batch_size}")
    print(f"  input_ids shape: {forward_batch.input_ids.shape}")
    print(f"  positions: {forward_batch.positions.tolist()}")
    print(f"  seq_lens: {batch.seq_lens.tolist()}")
    print(f"  prefix_lens: {batch.prefix_lens}")
    print(f"  extend_lens: {batch.extend_lens}")

  # Prefill
  out = model_runner.model.forward(
    forward_batch.input_ids,
    forward_batch.positions,
    forward_batch,
  )

  if verbose:
    _check_kv_cache_layout(kv_mgr, batch, page_size, "batch_prefill")

  # Extract per-request logits from prefill
  # For extend mode, logits has shape [total_tokens, vocab_size]
  # We need to get the last token's logits for each request
  all_logits = []
  all_tokens = []

  # Calculate where each sequence ends in the output
  extend_lens = [int(x) for x in batch.extend_lens]
  cumsum = 0
  for i, extend_len in enumerate(extend_lens):
    # Last token's logits for each request
    logits_i = out.logits[cumsum + extend_len - 1].detach().clone()
    all_logits.append([logits_i])
    next_token = int(torch.argmax(logits_i).item())
    reqs[i].output_ids.append(next_token)
    all_tokens.append([next_token])
    if stop_token_ids and next_token in stop_token_ids:
      reqs[i].finished = True
    cumsum += extend_len

  # Decode steps
  for step in range(1, steps):
    _filter_finished_from_batch(batch)
    if len(batch.reqs) == 0:
      break

    kv_mgr.prepare_for_decode(batch)
    forward_batch = ForwardBatch.init_new(batch, attn_backend)

    # Enable decode debugging for first decode step (before init_forward_metadata)
    if verbose and step == 1:
      forward_batch.debug_decode = True

    attn_backend.init_forward_metadata(forward_batch)
    if verbose and step <= 3:
      print(f"\n[BATCH DECODE step {step}]")
      print(f"  input_ids: {forward_batch.input_ids.tolist()}")
      print(f"  positions: {forward_batch.positions.tolist()}")
      print(f"  seq_lens: {batch.seq_lens.tolist()}")
      _check_kv_cache_layout(kv_mgr, batch, page_size, f"batch_decode_step{step}")
      # Print attention metadata
      metadata = attn_backend.forward_metadata
      if metadata is not None:
        print(
          f"  metadata.cache_seqlens_int32: {metadata.cache_seqlens_int32.tolist() if metadata.cache_seqlens_int32 is not None else None}"
        )
        print(
          f"  metadata.block_table: {metadata.block_table.tolist() if metadata.block_table is not None else None}"
        )
        print(f"  metadata.max_seq_len_k: {metadata.max_seq_len_k}")

    out = model_runner.model.forward(
      forward_batch.input_ids,
      forward_batch.positions,
      forward_batch,
    )

    # For decode mode, output shape is [batch_size, vocab_size]
    for active_i, req in enumerate(batch.reqs):
      req_idx = req_to_idx[id(req)]
      logits_i = out.logits[active_i].detach().clone()
      all_logits[req_idx].append(logits_i)
      next_token = int(torch.argmax(logits_i).item())
      req.output_ids.append(next_token)
      all_tokens[req_idx].append(next_token)
      if stop_token_ids and next_token in stop_token_ids:
        req.finished = True
      if verbose and step <= 5:
        _, top5 = torch.topk(logits_i, 5)
        # Also print logits stats for debugging
        print(
          f"  Step {step} Req {req_idx} (active_i={active_i}): next_token={next_token}, top5={top5.tolist()}, logits_mean={logits_i.mean().item():.4f}, logits_std={logits_i.std().item():.4f}"
        )

  # Collect final results
  results = []
  for i, req in enumerate(reqs):
    seq_len = len(all_prompt_ids[i]) + len(req.output_ids)
    kv_locs = kv_mgr.request_pool.read(req.req_pool_idx, slice(0, seq_len)).detach().cpu().tolist()
    results.append((all_logits[i], all_tokens[i], kv_locs))

  return results


def main():
  parser = argparse.ArgumentParser(description="Multi-Batch KV cache consistency test")
  parser.add_argument("--model", required=True, help="Model path or HF model id")
  parser.add_argument(
    "--prompts",
    nargs="+",
    default=[
      "1 + 1 = ?",
      "介绍一下你自己？",
      "What is Python ?",
    ],
    help="Prompts to test (will be used to create batch)",
  )
  parser.add_argument("--steps", type=int, default=5, help="Number of decode steps")
  parser.add_argument("--page-size", type=int, default=256, help="KV cache page size")
  parser.add_argument("--verbose", action="store_true", help="Print detailed debug info")
  parser.add_argument("--compare-hf", action="store_true", help="Also compare against HF model")
  parser.add_argument(
    "--skip-teacher-forcing-check",
    action="store_true",
    help="Skip teacher-forcing MiniInfer vs HF logits alignment",
  )
  parser.add_argument(
    "--teacher-rtol",
    type=float,
    default=1e-2,
    help="Relative tolerance for teacher-forcing logits comparison",
  )
  parser.add_argument(
    "--teacher-atol",
    type=float,
    default=1e-2,
    help="Absolute tolerance for teacher-forcing logits comparison",
  )
  parser.add_argument(
    "--teacher-topk",
    type=int,
    default=5,
    help="Top-k tokens to print on teacher-forcing mismatch",
  )
  parser.add_argument(
    "--skip-recompute-check",
    action="store_true",
    help="Skip per-step full recompute vs KV decode check",
  )
  parser.add_argument(
    "--recompute-rtol",
    type=float,
    default=1e-2,
    help="Relative tolerance for recompute logits comparison",
  )
  parser.add_argument(
    "--recompute-atol",
    type=float,
    default=1e-2,
    help="Absolute tolerance for recompute logits comparison",
  )
  args = parser.parse_args()

  torch.set_grad_enabled(False)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Device: {device}")
  print(f"Testing with {len(args.prompts)} prompts, {args.steps} decode steps each")

  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

  use_chat_template = _should_apply_chat_template(tokenizer, args.model)

  # Encode all prompts
  all_prompt_ids = [_encode_prompt(tokenizer, args.model, p) for p in args.prompts]
  stop_token_ids = _resolve_stop_token_ids(tokenizer)
  max_prompt_len = max(len(ids) for ids in all_prompt_ids)
  max_seq_len = max_prompt_len + args.steps + 10

  page_size = args.page_size
  # Batch prefill/decode allocation is page-granular per request.
  kv_size = _required_kv_size_tokens(
    [len(ids) for ids in all_prompt_ids],
    args.steps,
    page_size,
    safety_pages=1,
  )

  cfg = EngineConfig(
    args.model,
    max_num_seqs=len(args.prompts),
    max_context_len=max_seq_len,
    page_size=page_size,
  )

  print(f"Max prompt length: {max_prompt_len}")
  print(f"Chat template: {'enabled(auto)' if use_chat_template else 'disabled'}")
  print(f"Stop token ids: {sorted(stop_token_ids)}")
  print(f"KV cache size: {kv_size} tokens")
  print(f"Page size: {page_size}")

  # ============ Run each request independently ============
  print("\n" + "=" * 60)
  print("PHASE 1: Running each request independently (batch_size=1)")
  print("=" * 60)

  single_results = []
  recompute_all_ok = True
  for i, prompt_ids in enumerate(all_prompt_ids):
    print(f"\n--- Request {i}: '{args.prompts[i][:30]}...' (len={len(prompt_ids)}) ---")

    # Create fresh KV cache for each single run
    kv_mgr_single = _build_kv_cache_manager(
      cfg,
      size=_required_kv_size_tokens(
        [len(prompt_ids)],
        args.steps,
        page_size,
        safety_pages=1,
      ),
      max_context_len=max_seq_len,
      max_requests=1,
      page_size=page_size,
      device=device,
    )

    model_runner = ModelRunner(cfg, kv_mgr_single, 0, [])
    attn_backend = FlashAttention2Backend(kv_mgr_single)

    logits, tokens, kv_locs = run_single_request(
      model_runner,
      kv_mgr_single,
      attn_backend,
      prompt_ids,
      args.steps,
      device,
      page_size,
      stop_token_ids=stop_token_ids,
      verbose=args.verbose,
    )

    single_results.append(
      {
        "logits": logits,
        "tokens": tokens,
        "kv_locs": kv_locs,
      }
    )

    decoded = tokenizer.decode(tokens)
    print(f"  Generated: {[_fmt_token(tokenizer, t) for t in tokens]}")
    print(f"  Decoded: '{decoded}'")

    if not args.skip_recompute_check:
      check_ok, check_details, recompute_tokens = _check_kv_decode_vs_full_recompute(
        model_runner=model_runner,
        kv_mgr=kv_mgr_single,
        attn_backend=attn_backend,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        kv_logits_list=logits,
        kv_tokens=tokens,
        device=device,
        rtol=args.recompute_rtol,
        atol=args.recompute_atol,
      )
      single_results[-1]["recompute_ok"] = check_ok
      single_results[-1]["recompute_tokens"] = recompute_tokens
      single_results[-1]["recompute_details"] = check_details
      print(f"  Recompute check: {'PASS' if check_ok else 'FAIL'}")
      if not check_ok:
        recompute_all_ok = False
        print(f"    First mismatch detail: {check_details[0]}")

  # ============ Run all requests in one batch ============
  print("\n" + "=" * 60)
  print(f"PHASE 2: Running all {len(args.prompts)} requests in one batch")
  print("=" * 60)

  # Create KV cache for batch run
  kv_mgr_batch = _build_kv_cache_manager(
    cfg,
    size=kv_size,
    max_context_len=max_seq_len,
    max_requests=len(args.prompts),
    page_size=page_size,
    device=device,
  )

  model_runner = ModelRunner(cfg, kv_mgr_batch, 0, [])
  attn_backend_batch = FlashAttention2Backend(kv_mgr_batch)

  batch_results = run_batch_requests(
    model_runner,
    kv_mgr_batch,
    attn_backend_batch,
    all_prompt_ids,
    args.steps,
    device,
    page_size,
    stop_token_ids=stop_token_ids,
    verbose=args.verbose,
  )

  for i, (logits, tokens, kv_locs) in enumerate(batch_results):
    decoded = tokenizer.decode(tokens)
    print(f"Request {i}: {[_fmt_token(tokenizer, t) for t in tokens]}")
    print(f"  Decoded: '{decoded}'")

  # ============ Compare results ============
  print("\n" + "=" * 60)
  print("PHASE 3: Comparing single vs batch results")
  print("=" * 60)

  all_match = True
  for i in range(len(args.prompts)):
    print(f"\n--- Request {i}: '{args.prompts[i][:30]}...' ---")

    single_tokens = single_results[i]["tokens"]
    batch_tokens = batch_results[i][1]

    tokens_match = single_tokens == batch_tokens
    print(f"  Token match: {tokens_match}")
    if not tokens_match:
      print(f"    Single: {single_tokens}")
      print(f"    Batch:  {batch_tokens}")
      all_match = False

      # Find first mismatch position
      mismatch_found = False
      for j, (st, bt) in enumerate(zip(single_tokens, batch_tokens)):
        if st != bt:
          mismatch_found = True
          print(
            f"    First mismatch at step {j}: single={_fmt_token(tokenizer, st)}, batch={_fmt_token(tokenizer, bt)}"
          )

          # Compare logits at this position
          single_logits = single_results[i]["logits"][j].float()
          batch_logits = batch_results[i][0][j].float()

          logit_diff = (single_logits - batch_logits).abs()
          print(
            f"    Logits diff: max={logit_diff.max().item():.4e}, mean={logit_diff.mean().item():.4e}"
          )

          # Top-5 comparison
          _, top5_single = torch.topk(single_logits, 5)
          _, top5_batch = torch.topk(batch_logits, 5)
          print(f"    Top-5 single: {[_fmt_token(tokenizer, t.item()) for t in top5_single]}")
          print(f"    Top-5 batch:  {[_fmt_token(tokenizer, t.item()) for t in top5_batch]}")
          break
      if not mismatch_found and len(single_tokens) != len(batch_tokens):
        print(f"    Length mismatch: single={len(single_tokens)} vs batch={len(batch_tokens)}")
    else:
      # Even if tokens match, check logits closeness
      for j in range(len(single_tokens)):
        single_logits = single_results[i]["logits"][j].float()
        batch_logits = batch_results[i][0][j].float()

        logit_diff = (single_logits - batch_logits).abs()
        close = torch.allclose(single_logits, batch_logits, rtol=1e-2, atol=1e-2)
        if not close:
          print(f"    Step {j} logits NOT close: max_diff={logit_diff.max().item():.4e}")
          all_match = False

  # ============ Summary ============
  print("\n" + "=" * 60)
  print("SUMMARY")
  print("=" * 60)

  if all_match:
    print("✓ All tests PASSED: Single and batch runs produce identical results")
  else:
    print("✗ Tests FAILED: Discrepancy between single and batch runs")
    print("\nPossible causes:")
    print("  1. KV cache slot allocation issue (different requests interfering)")
    print("  2. req_pool_indices mapping error")
    print("  3. Incorrect position encoding in batch mode")
    print("  4. out_cache_loc calculation error")
    print("\nRun with --verbose for detailed debugging info")

  if not args.skip_recompute_check:
    if recompute_all_ok:
      print("✓ Recompute check PASSED: KV decode matches full recompute step-by-step")
    else:
      print("✗ Recompute check FAILED: KV decode diverges from full recompute")

  # ============ Optional: Compare with HF ============
  if args.compare_hf:
    print("\n" + "=" * 60)
    print("PHASE 4: Comparing with HuggingFace reference")
    print("=" * 60)

    hf_model = AutoModelForCausalLM.from_pretrained(
      args.model,
      torch_dtype=torch.float16 if device == "cuda" else torch.float32,
      trust_remote_code=True,
    ).to(device)
    hf_model.eval()

    teacher_all_ok = True
    for i, prompt_ids in enumerate(all_prompt_ids):
      print(f"\n--- Request {i} ---")

      # HF manual greedy (same decoding semantics as MiniInfer argmax loop).
      hf_tokens, _ = _hf_greedy_tokens_and_logits(
        hf_model=hf_model,
        prompt_ids=prompt_ids,
        steps=args.steps,
        device=device,
        stop_token_ids=stop_token_ids,
      )

      single_tokens = single_results[i]["tokens"]
      batch_tokens = batch_results[i][1]

      print(f"  HF tokens:     {[_fmt_token(tokenizer, t) for t in hf_tokens]}")
      print(f"  Single tokens: {[_fmt_token(tokenizer, t) for t in single_tokens]}")
      print(f"  Batch tokens:  {[_fmt_token(tokenizer, t) for t in batch_tokens]}")

      if hf_tokens == single_tokens:
        print(f"  ✓ Single matches HF")
      else:
        print(f"  ✗ Single DIFFERS from HF")

      if hf_tokens == batch_tokens:
        print(f"  ✓ Batch matches HF")
      else:
        print(f"  ✗ Batch DIFFERS from HF")

      if not args.skip_teacher_forcing_check:
        # Fresh MiniInfer runner for deterministic teacher-forcing prefill checks.
        kv_mgr_tf = _build_kv_cache_manager(
          cfg,
          size=_required_kv_size_tokens(
            [len(prompt_ids)],
            len(hf_tokens),
            page_size,
            safety_pages=1,
          ),
          max_context_len=max_seq_len,
          max_requests=1,
          page_size=page_size,
          device=device,
        )
        model_runner_tf = ModelRunner(cfg, kv_mgr_tf, 0, [])
        attn_backend_tf = FlashAttention2Backend(kv_mgr_tf)

        tf_ok, tf_details = _teacher_forcing_align_miniinfer_vs_hf(
          model_runner=model_runner_tf,
          kv_mgr=kv_mgr_tf,
          attn_backend=attn_backend_tf,
          hf_model=hf_model,
          tokenizer=tokenizer,
          prompt_ids=prompt_ids,
          teacher_tokens=hf_tokens,
          device=device,
          rtol=args.teacher_rtol,
          atol=args.teacher_atol,
          topk=args.teacher_topk,
        )
        print(
          f"  Teacher-forcing align: {'PASS' if tf_ok else 'FAIL'} " f"(steps={len(hf_tokens)})"
        )
        if not tf_ok:
          teacher_all_ok = False
          print(f"    First mismatch detail: {tf_details[0]}")

    if not args.skip_teacher_forcing_check:
      print("\n" + "-" * 60)
      if teacher_all_ok:
        print("Teacher-forcing summary: PASS (MiniInfer logits align with HF on forced prefixes)")
      else:
        print("Teacher-forcing summary: FAIL (MiniInfer vs HF diverge under forced prefixes)")


if __name__ == "__main__":
  main()
