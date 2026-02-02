"""
KV cache consistency check.

Compare per-step logits/hidden_state between:
A) decode forward with KV cache
B) full prefill forward on (prompt + generated tokens), no cache reuse

Optional: compare against HF Qwen2 outputs, per-layer hidden states, and verify Q/K/V weight loading.
This is intended as a debugging script, not a CI unit test.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add repo paths for top-level imports used across the codebase.
ROOT = Path(__file__).resolve().parents[2]
MINIINFER_ROOT = ROOT / "miniinfer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MINIINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(MINIINFER_ROOT))

from config.engine.config import EngineConfig
from engine.model_runner import ModelRunner
from engine.scheduler_batch import Req, ScheduledBatch, ForwardBatch
from kvcache.kv_cache_manager import KVCacheManager
from layers.attention_backend.flashattention_backend import FlashAttention2Backend
from utils.sampling_params import SamplingParams


def _round_up_to_page(size: int, page_size: int) -> int:
    if size <= 0:
        return page_size
    pages = max(1, math.ceil(size / page_size))
    return pages * page_size


def _resolve_dtype(name: str | None, device: str) -> torch.dtype:
    if name is None:
        return torch.float16 if device == "cuda" else torch.float32
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _should_apply_chat_template(
    tokenizer, model_name: str, use_chat_template: str
) -> bool:
    if use_chat_template == "true":
        return hasattr(tokenizer, "apply_chat_template")
    if use_chat_template == "false":
        return False
    # auto
    if not hasattr(tokenizer, "apply_chat_template"):
        return False
    name = str(model_name).lower()
    return ("instruct" in name) or ("chat" in name)


def _encode_prompt(
    tokenizer,
    prompt: str,
    model_name: str,
    use_chat_template: str,
    system_prompt: str,
) -> List[int]:
    if _should_apply_chat_template(tokenizer, model_name, use_chat_template):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, torch.Tensor):
            return encoded.tolist()
        if isinstance(encoded, dict) and "input_ids" in encoded:
            ids = encoded["input_ids"]
            return ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return list(encoded)
    return tokenizer.encode(prompt)


def _build_kv_cache_manager(
    cfg: EngineConfig,
    size: int,
    max_context_len: int,
    max_requests: int,
    page_size: int,
    device: str,
    enable_prefix_cache: bool = False,
) -> KVCacheManager:
    return KVCacheManager(
        size=size,
        max_requests=max_requests,
        max_context_len=max_context_len,
        num_layers=cfg.hf_config.num_hidden_layers,
        num_heads=cfg.hf_config.num_key_value_heads,
        head_dim=cfg.hf_config.hidden_size // cfg.hf_config.num_attention_heads,
        device=device,
        enable_prefix_cache=enable_prefix_cache,
        page_size=page_size,
    )


def _forward_with_backend(
    model, attn_backend, forward_batch: ForwardBatch, return_hidden_states: bool = False
):
    attn_backend.init_forward_metadata(forward_batch)
    if return_hidden_states:
        try:
            return model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                return_hidden_states=True,
            )
        except TypeError as exc:
            if "return_hidden_states" not in str(exc):
                raise
            print(
                "Warning: model.forward does not support return_hidden_states; "
                "falling back to default forward."
            )
    return model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )


def _select_last_logits(logits: torch.Tensor, is_decode: bool) -> torch.Tensor:
    if logits is None:
        return None
    if logits.dim() == 3:
        return logits[0, -1]
    if is_decode:
        return logits[0]
    return logits[-1]


def _select_last_hidden(hidden: torch.Tensor, is_decode: bool) -> torch.Tensor:
    if hidden is None:
        return None
    if hidden.dim() == 3:
        return hidden[0, -1]
    if is_decode:
        return hidden[0]
    return hidden[-1]


def _collect_hidden_layers(
    hidden_states: List[torch.Tensor] | None,
    last_hidden_state: torch.Tensor | None,
) -> List[torch.Tensor] | None:
    if hidden_states is None:
        return None
    layers = list(hidden_states)
    if last_hidden_state is not None:
        layers.append(last_hidden_state)
    return layers


def _topk_ids_vals(logits: torch.Tensor, k: int) -> Tuple[List[int], List[float]]:
    if logits is None:
        return [], []
    k = min(k, logits.shape[-1])
    vals, ids = torch.topk(logits, k=k, dim=-1)
    return ids.tolist(), vals.tolist()


def _fmt_token(tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        text = ""
    return f"{int(token_id)}:{text!r}"


def _compare_tensor(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> bool:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = torch.abs(actual_f - expected_f)
    max_abs = torch.max(diff).item()
    mean_abs = torch.mean(diff).item()
    close = torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol)
    print(
        f"{name}: close={close} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
        f"shape={tuple(actual.shape)}"
    )
    return close


def _free_batch(kv_mgr: KVCacheManager, batch: ScheduledBatch) -> None:
    if batch is None:
        return
    if batch.out_cache_loc is not None:
        kv_mgr.token_allocator.free(batch.out_cache_loc)
    if batch.req_pool_indices is not None:
        kv_mgr.request_pool.free(batch.req_pool_indices.cpu().tolist())


def _compare_step(
    step_idx: int,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    hidden_a: torch.Tensor,
    hidden_b: torch.Tensor,
    tokenizer,
    topk: int,
    logit_rtol: float,
    logit_atol: float,
    hidden_rtol: float,
    hidden_atol: float,
    pos_a: int,
    pos_b: int,
    expected_pos: int,
    show_topk: bool,
) -> Tuple[bool, bool, bool, bool]:
    logits_a_f = logits_a.float()
    logits_b_f = logits_b.float()

    top1_a = int(torch.argmax(logits_a_f).item())
    top1_b = int(torch.argmax(logits_b_f).item())
    top1_match = top1_a == top1_b

    topk_a, _ = _topk_ids_vals(logits_a_f, topk)
    topk_b, _ = _topk_ids_vals(logits_b_f, topk)
    topk_overlap = len(set(topk_a) & set(topk_b))

    logit_max_abs = torch.max(torch.abs(logits_a_f - logits_b_f)).item()
    logit_mean_abs = torch.mean(torch.abs(logits_a_f - logits_b_f)).item()
    logits_close = torch.allclose(
        logits_a_f, logits_b_f, rtol=logit_rtol, atol=logit_atol
    )

    hidden_a_f = hidden_a.float()
    hidden_b_f = hidden_b.float()
    hidden_max_abs = torch.max(torch.abs(hidden_a_f - hidden_b_f)).item()
    hidden_mean_abs = torch.mean(torch.abs(hidden_a_f - hidden_b_f)).item()
    hidden_close = torch.allclose(
        hidden_a_f, hidden_b_f, rtol=hidden_rtol, atol=hidden_atol
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        hidden_a_f, hidden_b_f, dim=0
    ).item()

    pos_ok = (pos_a == expected_pos) and (pos_b == expected_pos)

    print(
        f"step {step_idx:02d} | pos A={pos_a} B={pos_b} expected={expected_pos} "
        f"| top1 {'OK' if top1_match else 'DIFF'} "
        f"| topk overlap {topk_overlap}/{topk} "
        f"| logit max/mean {logit_max_abs:.4e}/{logit_mean_abs:.4e} "
        f"| hidden cos {cos_sim:.6f} max/mean {hidden_max_abs:.4e}/{hidden_mean_abs:.4e} "
        f"| logits_close={logits_close} hidden_close={hidden_close} pos_ok={pos_ok}"
    )

    if show_topk or (not top1_match):
        topk_a_tokens = ", ".join(_fmt_token(tokenizer, t) for t in topk_a)
        topk_b_tokens = ", ".join(_fmt_token(tokenizer, t) for t in topk_b)
        print(f"  topk A: {topk_a_tokens}")
        print(f"  topk B: {topk_b_tokens}")

    if not top1_match:
        print(
            f"  top1 A={_fmt_token(tokenizer, top1_a)} "
            f"B={_fmt_token(tokenizer, top1_b)}"
        )

    return top1_match, logits_close, hidden_close, pos_ok


def _compare_against_hf(
    step_idx: int,
    logits_ref: torch.Tensor,
    logits_hf: torch.Tensor,
    hidden_ref: torch.Tensor,
    hidden_hf: torch.Tensor,
    tokenizer,
    topk: int,
    logit_rtol: float,
    logit_atol: float,
    hidden_rtol: float,
    hidden_atol: float,
    show_topk: bool,
) -> Tuple[bool, bool, bool]:
    logits_ref_f = logits_ref.float()
    logits_hf_f = logits_hf.float()
    hidden_ref_f = hidden_ref.float()
    hidden_hf_f = hidden_hf.float()

    top1_ref = int(torch.argmax(logits_ref_f).item())
    top1_hf = int(torch.argmax(logits_hf_f).item())
    top1_match = top1_ref == top1_hf

    topk_ref, _ = _topk_ids_vals(logits_ref_f, topk)
    topk_hf, _ = _topk_ids_vals(logits_hf_f, topk)
    topk_overlap = len(set(topk_ref) & set(topk_hf))

    logit_max_abs = torch.max(torch.abs(logits_ref_f - logits_hf_f)).item()
    logit_mean_abs = torch.mean(torch.abs(logits_ref_f - logits_hf_f)).item()
    logits_close = torch.allclose(
        logits_ref_f, logits_hf_f, rtol=logit_rtol, atol=logit_atol
    )

    hidden_max_abs = torch.max(torch.abs(hidden_ref_f - hidden_hf_f)).item()
    hidden_mean_abs = torch.mean(torch.abs(hidden_ref_f - hidden_hf_f)).item()
    hidden_close = torch.allclose(
        hidden_ref_f, hidden_hf_f, rtol=hidden_rtol, atol=hidden_atol
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        hidden_ref_f, hidden_hf_f, dim=0
    ).item()

    print(
        f"  [HF] step {step_idx:02d} | top1 {'OK' if top1_match else 'DIFF'} "
        f"| topk overlap {topk_overlap}/{topk} "
        f"| logit max/mean {logit_max_abs:.4e}/{logit_mean_abs:.4e} "
        f"| hidden cos {cos_sim:.6f} max/mean {hidden_max_abs:.4e}/{hidden_mean_abs:.4e} "
        f"| logits_close={logits_close} hidden_close={hidden_close}"
    )

    if show_topk or (not top1_match):
        topk_ref_tokens = ", ".join(_fmt_token(tokenizer, t) for t in topk_ref)
        topk_hf_tokens = ", ".join(_fmt_token(tokenizer, t) for t in topk_hf)
        print(f"    topk ours: {topk_ref_tokens}")
        print(f"    topk  hf: {topk_hf_tokens}")

    if not top1_match:
        print(
            f"    top1 ours={_fmt_token(tokenizer, top1_ref)} "
            f"hf={_fmt_token(tokenizer, top1_hf)}"
        )

    return top1_match, logits_close, hidden_close


def _last_token_vec(hidden: torch.Tensor) -> torch.Tensor:
    if hidden is None:
        return None
    if hidden.dim() == 2:
        return hidden[-1]
    if hidden.dim() == 3:
        return hidden[0, -1]
    return hidden


def _alignment_score(ours: List[torch.Tensor], hf: List[torch.Tensor]) -> float:
    # Lower is better.
    scores = []
    for h_ours, h_hf in zip(ours, hf):
        ours_last = _last_token_vec(h_ours)
        hf_last = _last_token_vec(h_hf)
        diff = torch.abs(ours_last.float() - hf_last.float())
        scores.append(torch.mean(diff).item())
    return float(sum(scores) / max(len(scores), 1))


def _align_hidden_states(
    ours: List[torch.Tensor], hf: List[torch.Tensor]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], str]:
    if ours is None or hf is None:
        return [], [], "missing"
    if len(ours) == len(hf):
        return ours, hf, "exact"

    # If length differs by 1, try dropping embed or last and pick the better score.
    if abs(len(ours) - len(hf)) == 1:
        candidates = []
        if len(ours) > len(hf):
            candidates.append((ours[1:], hf, "drop_ours_embed"))
            candidates.append((ours[:-1], hf, "drop_ours_last"))
        else:
            candidates.append((ours, hf[1:], "drop_hf_embed"))
            candidates.append((ours, hf[:-1], "drop_hf_last"))

        scored = []
        for o_list, h_list, mode in candidates:
            score = _alignment_score(o_list, h_list)
            scored.append((score, o_list, h_list, mode))
        scored.sort(key=lambda x: x[0])
        _, best_o, best_h, best_mode = scored[0]
        return best_o, best_h, best_mode

    # Fallback: align by min length
    min_len = min(len(ours), len(hf))
    return ours[:min_len], hf[:min_len], "truncate"


def _compare_hidden_layers(
    step_idx: int,
    ours: List[torch.Tensor],
    hf: List[torch.Tensor],
    rtol: float,
    atol: float,
    label: str = "HF",
) -> bool:
    if ours is None or hf is None:
        print(
            f"  [{label}] hidden layer compare skipped (missing hidden_states). "
            f"ours={0 if ours is None else len(ours)} hf={0 if hf is None else len(hf)}"
        )
        return False

    ours_aligned, hf_aligned, mode = _align_hidden_states(ours, hf)
    if not ours_aligned or not hf_aligned:
        print(
            f"  [{label}] hidden layer compare skipped (missing hidden_states). "
            f"ours={len(ours)} hf={len(hf)}"
        )
        return False
    print(
        f"  [{label}] hidden layer compare: mode={mode} ours={len(ours)} hf={len(hf)} "
        f"aligned={len(ours_aligned)}"
    )
    all_ok = True
    for idx, (h_ours, h_hf) in enumerate(zip(ours_aligned, hf_aligned)):
        h_ours_last = _last_token_vec(h_ours)
        h_hf_last = _last_token_vec(h_hf)
        diff = torch.abs(h_ours_last.float() - h_hf_last.float())
        max_abs = torch.max(diff).item()
        mean_abs = torch.mean(diff).item()
        close = torch.allclose(
            h_ours_last.float(), h_hf_last.float(), rtol=rtol, atol=atol
        )
        all_ok = all_ok and close
        print(
            f"    layer[{idx:02d}] close={close} max_abs={max_abs:.4e} "
            f"mean_abs={mean_abs:.4e}"
        )
    return all_ok


def _debug_print_attn_metadata(label: str, forward_batch: ForwardBatch) -> None:
    backend = forward_batch.attn_backend
    md = getattr(backend, "forward_metadata", None)
    if md is None:
        print(f"[{label}] attn metadata: <missing>")
        return
    seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
    if forward_batch.positions is None:
        pos = None
    else:
        pos = forward_batch.positions.detach().cpu().tolist()
    out_loc = (
        None
        if forward_batch.out_cache_loc is None
        else forward_batch.out_cache_loc.detach().cpu().tolist()
    )
    cache_seqlens = (
        None
        if getattr(md, "cache_seqlens_int32", None) is None
        else md.cache_seqlens_int32.detach().cpu().tolist()
    )
    max_k = getattr(md, "max_seq_len_k", None)
    block_table = getattr(md, "block_table", None)
    block_shape = None if block_table is None else tuple(block_table.shape)
    print(
        f"[{label}] seq_lens={seq_lens} positions={pos} out_cache_loc={out_loc} "
        f"cache_seqlens={cache_seqlens} max_seq_len_k={max_k} block_table_shape={block_shape}"
    )


def _kv_loc_for_last_pos(kv_mgr: KVCacheManager, batch: ScheduledBatch) -> int:
    # After prepare_for_{extend,decode}, req_to_token_pool should be updated through last position.
    req_idx = int(batch.req_pool_indices[0].item())
    last_pos = int(batch.seq_lens[0].item() - 1)
    return int(kv_mgr.request_pool.req_to_token_pool()[req_idx, last_pos].item())


def _debug_compare_kv_last_token(
    step_idx: int,
    model,
    kv_mgr_a: KVCacheManager,
    batch_a: ScheduledBatch,
    kv_mgr_b: KVCacheManager,
    batch_b: ScheduledBatch,
    num_layers: int,
    rtol: float,
    atol: float,
    spy_a: dict | None = None,
    spy_b: dict | None = None,
) -> None:
    if torch.cuda.is_available():
        # Make sure KV writes from prior kernels are visible before we read buffers for debugging.
        torch.cuda.synchronize()

    loc_a = _kv_loc_for_last_pos(kv_mgr_a, batch_a)
    loc_b = _kv_loc_for_last_pos(kv_mgr_b, batch_b)
    out_a = int(batch_a.out_cache_loc[0].item())
    # For decode, out_cache_loc should be the last position.
    if out_a != loc_a:
        print(
            f"[KV] step {step_idx:02d} decode loc mismatch: out_cache_loc={out_a} "
            f"req_to_token_pool[last]={loc_a}"
        )
    token_a = (
        int(batch_a.input_ids[0].item()) if batch_a.input_ids is not None else None
    )
    token_b = (
        int(batch_b.input_ids[-1].item()) if batch_b.input_ids is not None else None
    )
    print(
        f"[KV] step {step_idx:02d} last token ids: A(decode)={token_a} B(prefill_last)={token_b} "
        f"loc_a={loc_a} loc_b={loc_b}"
    )

    for layer_id in range(num_layers):
        k_a, v_a = kv_mgr_a.get_kv_buffer(layer_id)
        k_b, v_b = kv_mgr_b.get_kv_buffer(layer_id)
        k_a_t = k_a[loc_a].float()
        v_a_t = v_a[loc_a].float()
        k_b_t = k_b[loc_b].float()
        v_b_t = v_b[loc_b].float()

        k_diff = (k_a_t - k_b_t).abs()
        v_diff = (v_a_t - v_b_t).abs()
        k_close = torch.allclose(k_a_t, k_b_t, rtol=rtol, atol=atol)
        v_close = torch.allclose(v_a_t, v_b_t, rtol=rtol, atol=atol)
        if not (k_close and v_close):
            print(
                f"[KV] step {step_idx:02d} layer[{layer_id:02d}] k_absmax/mean(A,B)="
                f"{k_a_t.abs().max().item():.4e}/{k_a_t.abs().mean().item():.4e},"
                f"{k_b_t.abs().max().item():.4e}/{k_b_t.abs().mean().item():.4e} "
                f"v_absmax/mean(A,B)="
                f"{v_a_t.abs().max().item():.4e}/{v_a_t.abs().mean().item():.4e},"
                f"{v_b_t.abs().max().item():.4e}/{v_b_t.abs().mean().item():.4e}"
            )
            if (
                spy_a is not None
                and spy_b is not None
                and layer_id in spy_a
                and layer_id in spy_b
            ):
                loc_a_in, k_a_in, v_a_in = spy_a[layer_id]
                loc_b_in, k_b_in, v_b_in = spy_b[layer_id]
                print(
                    f"[KV][in] layer[{layer_id:02d}] loc_in(A,B)={loc_a_in},{loc_b_in} "
                    f"k_absmax/mean(A,B)={k_a_in.abs().max().item():.4e}/{k_a_in.abs().mean().item():.4e},"
                    f"{k_b_in.abs().max().item():.4e}/{k_b_in.abs().mean().item():.4e} "
                    f"v_absmax/mean(A,B)={v_a_in.abs().max().item():.4e}/{v_a_in.abs().mean().item():.4e},"
                    f"{v_b_in.abs().max().item():.4e}/{v_b_in.abs().mean().item():.4e}"
                )
                k_in_close = torch.allclose(
                    k_a_in.float(), k_b_in.float(), rtol=rtol, atol=atol
                )
                v_in_close = torch.allclose(
                    v_a_in.float(), v_b_in.float(), rtol=rtol, atol=atol
                )
                print(
                    f"[KV][in] layer[{layer_id:02d}] k_in_close={k_in_close} v_in_close={v_in_close}"
                )
                # Compare what we tried to write vs what ended up in storage.
                k_a_store_close = torch.allclose(
                    k_a_in.float(), k_a_t.float(), rtol=rtol, atol=atol
                )
                v_a_store_close = torch.allclose(
                    v_a_in.float(), v_a_t.float(), rtol=rtol, atol=atol
                )
                k_b_store_close = torch.allclose(
                    k_b_in.float(), k_b_t.float(), rtol=rtol, atol=atol
                )
                v_b_store_close = torch.allclose(
                    v_b_in.float(), v_b_t.float(), rtol=rtol, atol=atol
                )
                print(
                    f"[KV][store] layer[{layer_id:02d}] A(in==store) k={k_a_store_close} v={v_a_store_close} "
                    f"B(in==store) k={k_b_store_close} v={v_b_store_close}"
                )
            print(
                f"[KV] step {step_idx:02d} first mismatch at layer[{layer_id:02d}] "
                f"k_close={k_close} k_max/mean={k_diff.max().item():.4e}/{k_diff.mean().item():.4e} "
                f"v_close={v_close} v_max/mean={v_diff.max().item():.4e}/{v_diff.mean().item():.4e}"
            )

            # For the first mismatch, compute a reference K/V for layer0 from pure math
            # (embedding -> RMSNorm -> qkv_proj -> RoPE) and see which side matches.
            if layer_id == 0 and model is not None and token_a is not None:
                try:
                    # Access fused Qwen2 modules.
                    block0 = model.qwen2.layers[0]
                    attn0 = block0.self_attn
                    ln0 = block0.input_layernorm
                    device = k_a.device
                    tok = torch.tensor([token_a], device=device, dtype=torch.int64)
                    pos = torch.tensor(
                        [int(batch_a.seq_lens[0].item() - 1)],
                        device=device,
                        dtype=torch.int64,
                    )
                    x = model.qwen2.embedding(tok)  # [1, hidden]
                    x = ln0(x)  # [1, hidden]
                    qkv = attn0.qkv_proj(x)
                    q_ref, k_ref, v_ref = qkv.split(
                        [attn0.q_size, attn0.kv_size, attn0.kv_size], dim=-1
                    )
                    q_ref, k_ref = attn0.rope(pos, q_ref, k_ref)
                    k_ref = k_ref.view(-1, attn0.num_kv_heads, attn0.head_dim)[
                        0
                    ].float()
                    v_ref = v_ref.view(-1, attn0.num_kv_heads, attn0.head_dim)[
                        0
                    ].float()

                    def _cmp(name: str, a: torch.Tensor, b: torch.Tensor):
                        d = (a - b).abs()
                        return (
                            f"{name} close={torch.allclose(a,b,rtol=rtol,atol=atol)} "
                            f"max/mean={d.max().item():.4e}/{d.mean().item():.4e}"
                        )

                    print(f"[KV][ref] pos={int(pos.item())} token={token_a}")
                    print(
                        f"[KV][ref] k_ref_absmax/mean={k_ref.abs().max().item():.4e}/{k_ref.abs().mean().item():.4e} "
                        f"v_ref_absmax/mean={v_ref.abs().max().item():.4e}/{v_ref.abs().mean().item():.4e}"
                    )
                    print(f"[KV][ref] {_cmp('k_ref vs A', k_ref, k_a_t)}")
                    print(f"[KV][ref] {_cmp('k_ref vs B', k_ref, k_b_t)}")
                    print(f"[KV][ref] {_cmp('v_ref vs A', v_ref, v_a_t)}")
                    print(f"[KV][ref] {_cmp('v_ref vs B', v_ref, v_b_t)}")
                except Exception as exc:
                    print(f"[KV][ref] skipped due to error: {exc}")
            return
    print(f"[KV] step {step_idx:02d} last-token KV match across all layers (a vs b).")


def _debug_check_req_to_token_pool(
    kv_mgr: KVCacheManager,
    batch: ScheduledBatch,
    page_size: int,
    stage: str,
) -> bool:
    if batch is None or batch.req_pool_indices is None or batch.seq_lens is None:
        return True
    seq_lens = (
        batch.seq_lens_cpu.tolist()
        if batch.seq_lens_cpu is not None
        else batch.seq_lens.detach().cpu().tolist()
    )
    req_indices = batch.req_pool_indices.detach().cpu().tolist()
    issues = 0

    # Precompute last locs for extend
    extend_last_locs = None
    if stage == "extend" and batch.extend_lens and batch.out_cache_loc is not None:
        extend_last_locs = []
        offset = 0
        for extend_len in batch.extend_lens:
            if extend_len > 0:
                last_loc = int(batch.out_cache_loc[offset + extend_len - 1].item())
            else:
                last_loc = None
            extend_last_locs.append(last_loc)
            offset += int(extend_len)

    for i, req_idx in enumerate(req_indices):
        seq_len = int(seq_lens[i])
        kv_locs = (
            kv_mgr.request_pool.read(req_idx, slice(0, seq_len)).detach().cpu().tolist()
        )
        if any(loc == 0 for loc in kv_locs):
            print(f"[map][{stage}] req_idx={req_idx} has zero entries")
            issues += 1

        # Continuity within pages
        for j in range(1, len(kv_locs)):
            prev = kv_locs[j - 1]
            cur = kv_locs[j]
            if prev == 0 or cur == 0:
                continue
            prev_page = prev // page_size
            cur_page = cur // page_size
            if cur_page == prev_page:
                if cur != prev + 1:
                    print(
                        f"[map][{stage}] req_idx={req_idx} break at pos {j}: "
                        f"{prev}->{cur} (same page)"
                    )
                    issues += 1
                    break
            elif cur_page == prev_page + 1:
                if (prev % page_size) != (page_size - 1) or (cur % page_size) != 0:
                    print(
                        f"[map][{stage}] req_idx={req_idx} break at pos {j}: "
                        f"{prev}->{cur} (page boundary)"
                    )
                    issues += 1
                    break
            else:
                print(
                    f"[map][{stage}] req_idx={req_idx} jump at pos {j}: "
                    f"{prev}->{cur} (page {prev_page}->{cur_page})"
                )
                issues += 1
                break

        if stage == "decode" and batch.out_cache_loc is not None:
            expected = int(batch.out_cache_loc[i].item())
            if kv_locs and kv_locs[-1] != expected:
                print(
                    f"[map][decode] req_idx={req_idx} last_loc mismatch: "
                    f"{kv_locs[-1]} != {expected}"
                )
                issues += 1
        if stage == "extend" and extend_last_locs is not None:
            expected = extend_last_locs[i]
            if expected is not None and kv_locs and kv_locs[-1] != expected:
                print(
                    f"[map][extend] req_idx={req_idx} last_loc mismatch: "
                    f"{kv_locs[-1]} != {expected}"
                )
                issues += 1

    print(f"[map][{stage}] checked {len(req_indices)} reqs, issues={issues}")
    return issues == 0


def main():
    parser = argparse.ArgumentParser(description="KV cache consistency check")
    parser.add_argument("--model", required=True, help="Model path or HF model id")
    parser.add_argument("--prompt", default="What is Qwen?", help="Prompt text")
    parser.add_argument("--steps", type=int, default=8, help="Number of decode steps")
    parser.add_argument("--topk", type=int, default=5, help="Top-k to compare")
    parser.add_argument("--page-size", type=int, default=256, help="KV cache page size")
    parser.add_argument(
        "--compare-hf",
        action="store_true",
        help="Compare our full-prefill outputs against HF model",
    )
    parser.add_argument(
        "--compare-hf-layers",
        action="store_true",
        help="Compare per-layer hidden states against HF model (uses B path)",
    )

    parser.add_argument(
        "--no-debug-mapping",
        action="store_true",
        help="Disable debug: check req_to_token_pool mapping continuity",
    )
    parser.add_argument(
        "--debug-decode-past-len-cache",
        action="store_true",
        help=(
            "Debug only: set attention cache_seqlens = (seq_lens - 1) in decode. "
            "This typically makes decode differ from full prefill because the current token KV "
            "is written before attention and should be counted."
        ),
    )
    parser.add_argument(
        "--no-debug-compare-ab-layers",
        action="store_true",
        help="Disable debug: compare A(decode) vs B(prefill) per-layer hidden states",
    )
    parser.add_argument(
        "--debug-print-attn-metadata",
        action="store_true",
        help="Debug: print decode/prefill forward_batch + attention backend metadata each step",
    )
    parser.add_argument(
        "--debug-compare-kv",
        action="store_true",
        help=(
            "Debug: compare KV cache values for the last token between A(decode) and B(prefill) "
            "for the first mismatching layer."
        ),
    )
    parser.add_argument(
        "--hf-dtype",
        default=None,
        help="HF model dtype: fp16|bf16|fp32 (default: fp16 on cuda, fp32 on cpu)",
    )
    parser.add_argument(
        "--check-kv-weights",
        action="store_true",
        help="Compare HF Q/K/V weights with our fused QKV weights",
    )
    parser.add_argument("--weight-rtol", type=float, default=None)
    parser.add_argument("--weight-atol", type=float, default=None)
    parser.add_argument(
        "--use-chat-template",
        choices=["auto", "true", "false"],
        default="auto",
        help="Use tokenizer.apply_chat_template if available",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt for chat template",
    )
    parser.add_argument("--logit-rtol", type=float, default=0.1)
    parser.add_argument("--logit-atol", type=float, default=0.1)
    parser.add_argument("--hidden-rtol", type=float, default=0.1)
    parser.add_argument("--hidden-atol", type=float, default=0.1)
    parser.add_argument("--strict", action="store_true", help="Raise on any mismatch")
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first mismatch"
    )
    parser.add_argument(
        "--show-topk", action="store_true", help="Print top-k tokens each step"
    )
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be >= 1")

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        trust_remote_code=True,
    )

    prompt_ids = _encode_prompt(
        tokenizer=tokenizer,
        prompt=args.prompt,
        model_name=args.model,
        use_chat_template=args.use_chat_template,
        system_prompt=args.system_prompt,
    )
    prompt_len = len(prompt_ids)
    max_seq_len = prompt_len + args.steps
    page_size = int(args.page_size)
    kv_size = _round_up_to_page(max_seq_len, page_size)

    cfg = EngineConfig(
        args.model,
        max_num_seqs=1,
        max_context_len=max_seq_len,
        max_total_tokens=kv_size,
    )

    if max_seq_len > cfg.max_context_len:
        raise ValueError(
            f"Sequence length {max_seq_len} exceeds model max_context_len {cfg.max_context_len}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CPU detected. FlashAttention backends may not be available.")

    debug_decode_past_len = bool(getattr(args, "debug_decode_past_len_cache", False))
    debug_mapping = not getattr(args, "no_debug_mapping", False)
    debug_compare_ab_layers = not getattr(args, "no_debug_compare_ab_layers", False)

    kv_mgr_a = _build_kv_cache_manager(
        cfg,
        size=kv_size,
        max_context_len=max_seq_len,
        max_requests=1,
        page_size=page_size,
        device=device,
        enable_prefix_cache=False,
    )

    kv_mgr_b = _build_kv_cache_manager(
        cfg,
        size=kv_size,
        max_context_len=max_seq_len,
        max_requests=1,
        page_size=page_size,
        device=device,
        enable_prefix_cache=False,
    )

    kv_spy_a = {}
    kv_spy_b = {}
    act_spy = {"tag": None, "emb": {}, "ln0": {}, "qkv0": {}}
    if args.debug_compare_kv:
        # Record the per-layer KV tensor that the model *attempts* to write for the last token,
        # so we can distinguish "compute differs" vs "storage differs" vs "flash-attn overwrites".
        orig_set_a = kv_mgr_a.set_kv_buffer
        orig_set_b = kv_mgr_b.set_kv_buffer

        def _make_spy(orig_fn, spy_dict):
            def _spy(
                layer_id: int,
                loc: torch.Tensor,
                cache_k: torch.Tensor,
                cache_v: torch.Tensor,
            ):
                try:
                    if loc is not None and cache_k is not None and cache_v is not None:
                        idx = -1 if loc.numel() > 1 else 0
                        loc_i = int(loc[idx].item())
                        k_i = cache_k[idx].detach().clone()
                        v_i = cache_v[idx].detach().clone()
                        spy_dict[int(layer_id)] = (loc_i, k_i, v_i)
                except Exception as exc:
                    print(f"[KV][spy] failed to record layer[{layer_id}]: {exc}")
                return orig_fn(layer_id, loc, cache_k, cache_v)

            return _spy

        kv_mgr_a.set_kv_buffer = _make_spy(orig_set_a, kv_spy_a)
        kv_mgr_b.set_kv_buffer = _make_spy(orig_set_b, kv_spy_b)

    model_runner = ModelRunner(cfg, kv_mgr_a, 0, [])

    if args.debug_compare_kv:
        # Capture pre-attention activations for layer0: embedding -> input_layernorm -> qkv_proj.
        emb = model_runner.model.qwen2.embedding
        ln0 = model_runner.model.qwen2.layers[0].input_layernorm
        qkv0 = model_runner.model.qwen2.layers[0].self_attn.qkv_proj

        orig_emb_fwd = emb.forward
        orig_ln0_fwd = ln0.forward
        orig_qkv0_fwd = qkv0.forward

        def _tag():
            return act_spy.get("tag", None)

        def _emb_fwd(x):
            y = orig_emb_fwd(x)
            tag = _tag()
            if tag is not None:
                try:
                    act_spy["emb"][tag] = y[-1].detach().clone()
                except Exception as exc:
                    print(f"[ACT][emb] capture failed: {exc}")
            return y

        def _ln0_fwd(x, residual=None):
            y = orig_ln0_fwd(x, residual=residual)
            tag = _tag()
            if tag is not None and residual is None:
                try:
                    act_spy["ln0"][tag] = y[-1].detach().clone()
                except Exception as exc:
                    print(f"[ACT][ln0] capture failed: {exc}")
            return y

        def _qkv0_fwd(x):
            y = orig_qkv0_fwd(x)
            tag = _tag()
            if tag is not None:
                try:
                    act_spy["qkv0"][tag] = y[-1].detach().clone()
                except Exception as exc:
                    print(f"[ACT][qkv0] capture failed: {exc}")
            return y

        emb.forward = _emb_fwd
        ln0.forward = _ln0_fwd
        qkv0.forward = _qkv0_fwd

    attn_backend_a = model_runner.attn_backend
    attn_backend_b = FlashAttention2Backend(kv_mgr_b)

    hf_model = None
    hf_dtype = None
    if args.compare_hf or args.check_kv_weights or args.compare_hf_layers:
        hf_dtype = _resolve_dtype(args.hf_dtype, device)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=hf_dtype,
            trust_remote_code=True,
        ).to(device)
        hf_model.eval()
        if args.compare_hf or args.compare_hf_layers:
            print(f"[HF] compare enabled, dtype={hf_dtype}")

    # Prefill A: prompt only, then sample the first token (token_0).
    req_a = Req(prompt_ids, SamplingParams(temperature=0, max_tokens=args.steps + 1))
    kv_mgr_a.prefix_for_waiting_req(req_a)
    batch_a = ScheduledBatch.init_new([req_a], device=model_runner.device)
    kv_mgr_a.prepare_for_extend(batch_a)

    forward_batch_a = ForwardBatch.init_new(batch_a, attn_backend_a)
    forward_batch_a.debug_use_past_len_cache = debug_decode_past_len
    if debug_mapping:
        _debug_check_req_to_token_pool(
            kv_mgr_a, batch_a, page_size=page_size, stage="extend"
        )
    out_prefill = _forward_with_backend(
        model_runner.model, attn_backend_a, forward_batch_a
    )

    logits_prefill = _select_last_logits(out_prefill.logits, is_decode=False)
    hidden_prefill = _select_last_hidden(out_prefill.last_hidden_state, is_decode=False)

    next_token = int(torch.argmax(logits_prefill).item())
    req_a.output_ids.append(next_token)

    print(
        f"Prompt length: {prompt_len}, decode steps: {args.steps}, "
        f"KV size: {kv_size}, device: {device}, "
        f"chat_template={args.use_chat_template}"
    )
    print(f"Initial token_0: {_fmt_token(tokenizer, next_token)}")

    # Default tolerances by dtype.
    logit_rtol = args.logit_rtol
    logit_atol = args.logit_atol
    hidden_rtol = args.hidden_rtol
    hidden_atol = args.hidden_atol

    if logit_rtol is None or logit_atol is None:
        if logits_prefill.dtype == torch.float16:
            logit_rtol = 5.0e-2 if logit_rtol is None else logit_rtol
            logit_atol = 5.0e-4 if logit_atol is None else logit_atol
        else:
            logit_rtol = 1.0e-5 if logit_rtol is None else logit_rtol
            logit_atol = 1.0e-8 if logit_atol is None else logit_atol

    if hidden_rtol is None or hidden_atol is None:
        if hidden_prefill.dtype == torch.float16:
            hidden_rtol = 5.0e-2 if hidden_rtol is None else hidden_rtol
            hidden_atol = 5.0e-4 if hidden_atol is None else hidden_atol
        else:
            hidden_rtol = 1.0e-5 if hidden_rtol is None else hidden_rtol
            hidden_atol = 1.0e-8 if hidden_atol is None else hidden_atol

    mismatches = 0
    mismatches_hf = 0

    # Decode steps: compare A(decode) vs B(full prefill) for each step.
    for step_idx in range(1, args.steps + 1):
        # A: decode with KV cache.
        kv_mgr_a.prepare_for_decode(batch_a)
        forward_batch_a = ForwardBatch.init_new(batch_a, attn_backend_a)
        forward_batch_a.debug_use_past_len_cache = debug_decode_past_len
        if debug_mapping:
            _debug_check_req_to_token_pool(
                kv_mgr_a, batch_a, page_size=page_size, stage="decode"
            )
        if args.debug_compare_kv:
            act_spy["tag"] = "A"
        out_a = _forward_with_backend(
            model_runner.model,
            attn_backend_a,
            forward_batch_a,
            return_hidden_states=debug_compare_ab_layers,
        )
        if args.debug_compare_kv:
            act_spy["tag"] = None
        if args.debug_print_attn_metadata:
            _debug_print_attn_metadata("A/decode", forward_batch_a)

        logits_a = _select_last_logits(out_a.logits, is_decode=True)
        hidden_a = _select_last_hidden(out_a.last_hidden_state, is_decode=True)

        pos_a = int(forward_batch_a.positions.item())
        expected_pos = prompt_len + step_idx - 1

        # B: full prefill on prompt + generated tokens so far (no cache reuse).
        seq_ids = prompt_ids + req_a.output_ids
        req_b = Req(seq_ids, SamplingParams(temperature=0, max_tokens=1))
        kv_mgr_b.prefix_for_waiting_req(req_b)
        batch_b = ScheduledBatch.init_new([req_b], device=model_runner.device)
        kv_mgr_b.prepare_for_extend(batch_b)

        forward_batch_b = ForwardBatch.init_new(batch_b, attn_backend_b)
        forward_batch_b.debug_use_past_len_cache = debug_decode_past_len
        if debug_mapping:
            _debug_check_req_to_token_pool(
                kv_mgr_b, batch_b, page_size=page_size, stage="extend"
            )
        if args.debug_compare_kv:
            act_spy["tag"] = "B"
        out_b = _forward_with_backend(
            model_runner.model,
            attn_backend_b,
            forward_batch_b,
            return_hidden_states=(args.compare_hf_layers or debug_compare_ab_layers),
        )
        if args.debug_compare_kv:
            act_spy["tag"] = None
        if args.debug_print_attn_metadata:
            _debug_print_attn_metadata("B/prefill", forward_batch_b)

        logits_b = _select_last_logits(out_b.logits, is_decode=False)
        hidden_b = _select_last_hidden(out_b.last_hidden_state, is_decode=False)

        pos_b = int(forward_batch_b.positions[-1].item())

        top1_ok, logits_ok, hidden_ok, pos_ok = _compare_step(
            step_idx=step_idx,
            logits_a=logits_a,
            logits_b=logits_b,
            hidden_a=hidden_a,
            hidden_b=hidden_b,
            tokenizer=tokenizer,
            topk=args.topk,
            logit_rtol=logit_rtol,
            logit_atol=logit_atol,
            hidden_rtol=hidden_rtol,
            hidden_atol=hidden_atol,
            pos_a=pos_a,
            pos_b=pos_b,
            expected_pos=expected_pos,
            show_topk=args.show_topk,
        )

        hf_top1_ok = True
        hf_logits_ok = True
        hf_hidden_ok = True
        hf_layers_ok = True
        if args.compare_hf or args.compare_hf_layers:
            with torch.no_grad():
                hf_input_ids = torch.tensor(seq_ids, device=device).unsqueeze(0)
                hf_out = hf_model(
                    hf_input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hf_logits = hf_out.logits[0, -1]
                hf_hidden = hf_out.hidden_states[-1][0, -1]
            if args.compare_hf:
                hf_top1_ok, hf_logits_ok, hf_hidden_ok = _compare_against_hf(
                    step_idx=step_idx,
                    logits_ref=logits_b,
                    logits_hf=hf_logits,
                    hidden_ref=hidden_b,
                    hidden_hf=hf_hidden,
                    tokenizer=tokenizer,
                    topk=args.topk,
                    logit_rtol=logit_rtol,
                    logit_atol=logit_atol,
                    hidden_rtol=hidden_rtol,
                    hidden_atol=hidden_atol,
                    show_topk=args.show_topk,
                )
            if args.compare_hf_layers:
                ours_layers = _collect_hidden_layers(
                    out_b.hidden_states, out_b.last_hidden_state
                )
                hf_layers_ok = _compare_hidden_layers(
                    step_idx=step_idx,
                    ours=ours_layers,
                    hf=hf_out.hidden_states,
                    rtol=hidden_rtol,
                    atol=hidden_atol,
                    label="HF",
                )

        if debug_compare_ab_layers and out_a.hidden_states is not None:
            out_a_layers = _collect_hidden_layers(
                out_a.hidden_states, out_a.last_hidden_state
            )
            out_b_layers = _collect_hidden_layers(
                out_b.hidden_states, out_b.last_hidden_state
            )
            _compare_hidden_layers(
                step_idx=step_idx,
                ours=out_a_layers,
                hf=out_b_layers,
                rtol=hidden_rtol,
                atol=hidden_atol,
                label="AB",
            )
        if args.debug_compare_kv:
            _debug_compare_kv_last_token(
                step_idx=step_idx,
                model=model_runner.model,
                kv_mgr_a=kv_mgr_a,
                batch_a=batch_a,
                kv_mgr_b=kv_mgr_b,
                batch_b=batch_b,
                num_layers=cfg.hf_config.num_hidden_layers,
                rtol=hidden_rtol,
                atol=hidden_atol,
                spy_a=kv_spy_a,
                spy_b=kv_spy_b,
            )
            # Also compare the layer0 pre-attention activations.
            if "A" in act_spy["emb"] and "B" in act_spy["emb"]:
                a = act_spy["emb"]["A"].float()
                b = act_spy["emb"]["B"].float()
                d = (a - b).abs()
                print(
                    f"[ACT] emb last-token close={torch.allclose(a,b,rtol=hidden_rtol,atol=hidden_atol)} "
                    f"max/mean={d.max().item():.4e}/{d.mean().item():.4e}"
                )
            if "A" in act_spy["ln0"] and "B" in act_spy["ln0"]:
                a = act_spy["ln0"]["A"].float()
                b = act_spy["ln0"]["B"].float()
                d = (a - b).abs()
                print(
                    f"[ACT] ln0 last-token close={torch.allclose(a,b,rtol=hidden_rtol,atol=hidden_atol)} "
                    f"max/mean={d.max().item():.4e}/{d.mean().item():.4e}"
                )
            if "A" in act_spy["qkv0"] and "B" in act_spy["qkv0"]:
                a = act_spy["qkv0"]["A"].float()
                b = act_spy["qkv0"]["B"].float()
                d = (a - b).abs()
                print(
                    f"[ACT] qkv0 last-token close={torch.allclose(a,b,rtol=hidden_rtol,atol=hidden_atol)} "
                    f"max/mean={d.max().item():.4e}/{d.mean().item():.4e}"
                )

        # Free B allocations to keep KV usage bounded.
        _free_batch(kv_mgr_b, batch_b)

        if not (top1_ok and logits_ok and hidden_ok and pos_ok):
            mismatches += 1
            if args.fail_fast:
                break

        if args.compare_hf and not (hf_top1_ok and hf_logits_ok and hf_hidden_ok):
            mismatches_hf += 1
            if args.fail_fast:
                break
        if args.compare_hf_layers and not hf_layers_ok:
            mismatches_hf += 1
            if args.fail_fast:
                break

        # Sample next token from A for the next step.
        next_token = int(torch.argmax(logits_a).item())
        req_a.output_ids.append(next_token)
        if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
            print("EOS hit; stopping early.")
            break

    print(f"Done. mismatches={mismatches} hf_mismatches={mismatches_hf}")
    if args.strict and (mismatches > 0 or mismatches_hf > 0):
        raise AssertionError(f"Found mismatches: kv={mismatches}, hf={mismatches_hf}")


if __name__ == "__main__":
    main()
