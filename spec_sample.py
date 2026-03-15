from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch


class TreeMaskMode(IntEnum):
  FULL_MASK = 0
  QLEN_ONLY = 1
  QLEN_ONLY_BITPACKING = 2


@dataclass
class DraftTree:
  draft_tokens: torch.Tensor
  tree_mask: torch.Tensor
  positions: torch.Tensor
  retrive_index: torch.Tensor
  retrive_next_token: torch.Tensor
  retrive_next_sibling: torch.Tensor
  parent_local: torch.Tensor
  depth: torch.Tensor


def _as_mode(mode: TreeMaskMode | int) -> TreeMaskMode:
  if isinstance(mode, TreeMaskMode):
    return mode
  return TreeMaskMode(int(mode))


def _pack_mask_bits(mask: torch.Tensor) -> torch.Tensor:
  if mask.dtype != torch.bool:
    raise TypeError("bitpacking expects a bool mask")
  width = mask.shape[-1]
  pad = (-width) % 8
  if pad:
    mask = torch.nn.functional.pad(mask, (0, pad))
  mask_u8 = mask.to(torch.uint8).reshape(*mask.shape[:-1], -1, 8)
  shifts = torch.arange(8, device=mask.device, dtype=torch.uint8)
  return torch.sum(mask_u8 << shifts, dim=-1, dtype=torch.uint8)


def _build_tree_mask_from_parent(parent_local: torch.Tensor) -> torch.Tensor:
  bs, num_nodes = parent_local.shape
  mask = torch.zeros((bs, num_nodes, num_nodes), dtype=torch.bool, device=parent_local.device)
  for b in range(bs):
    for node in range(num_nodes):
      cur = node
      while cur >= 0:
        mask[b, node, cur] = True
        cur = int(parent_local[b, cur].item())
  return mask


def build_draft_tree(
  verified_id: torch.Tensor,
  parent_list: torch.Tensor,
  top_scores_index: torch.Tensor,
  draft_tokens: torch.Tensor,
  seq_lens: torch.Tensor,
  seq_lens_sum: Optional[int] = None,
  topk: int = 1,
  spec_steps: int = 1,
  num_verify_tokens: Optional[int] = None,
  tree_mask_mode: TreeMaskMode | int = TreeMaskMode.QLEN_ONLY,
  tree_mask_buf: Optional[torch.Tensor] = None,
) -> DraftTree:
  """
  Python reference for tree packing used by speculative verify.

  Assumptions:
  - `verified_id` is the root token and is materialized at local slot 0.
  - real bank index 0 denotes the root.
  - `top_scores_index` is prefix-closed, so every selected node's parent is also selected
    (or is the root).
  """
  if verified_id.ndim != 1:
    raise ValueError("verified_id must have shape [bs]")
  if draft_tokens.shape != top_scores_index.shape:
    raise ValueError(
      "draft_tokens and top_scores_index must share shape [bs, num_verify_tokens - 1]"
    )
  if seq_lens.shape != verified_id.shape:
    raise ValueError("seq_lens must have shape [bs]")
  if topk <= 0:
    raise ValueError("topk must be positive")
  if spec_steps <= 0:
    raise ValueError("spec_steps must be positive")

  bs = verified_id.shape[0]
  selected_non_root = draft_tokens.shape[1]
  if num_verify_tokens is None:
    num_verify_tokens = selected_non_root + 1
  if num_verify_tokens != selected_non_root + 1:
    raise ValueError("num_verify_tokens must equal draft_tokens.shape[1] + 1")

  device = verified_id.device
  flat_tokens = torch.empty((bs, num_verify_tokens), dtype=draft_tokens.dtype, device=device)
  flat_tokens[:, 0] = verified_id.to(draft_tokens.dtype)
  flat_tokens[:, 1:] = draft_tokens

  parent_local = torch.full((bs, num_verify_tokens), -1, dtype=torch.int64, device=device)
  retrive_next_token = torch.full((bs, num_verify_tokens), -1, dtype=torch.int64, device=device)
  retrive_next_sibling = torch.full((bs, num_verify_tokens), -1, dtype=torch.int64, device=device)
  depth = torch.zeros((bs, num_verify_tokens), dtype=torch.int64, device=device)
  selected_bank = torch.empty((bs, num_verify_tokens), dtype=torch.int64, device=device)
  selected_bank[:, 0] = 0
  selected_bank[:, 1:] = top_scores_index.to(torch.int64)

  for b in range(bs):
    bank_to_local = {0: 0}
    for local in range(1, num_verify_tokens):
      bank_idx = int(selected_bank[b, local].item())
      if bank_idx in bank_to_local:
        raise ValueError(f"duplicate selected bank index {bank_idx} in batch {b}")
      bank_to_local[bank_idx] = local

    children: list[list[tuple[int, int]]] = [[] for _ in range(num_verify_tokens)]
    for local in range(1, num_verify_tokens):
      bank_idx = int(selected_bank[b, local].item())
      parent_slot = bank_idx // topk
      if parent_slot < 0 or parent_slot >= parent_list.shape[1]:
        raise IndexError(
          f"parent_slot={parent_slot} out of range for batch {b}; "
          f"bank_idx={bank_idx}, parent_list.shape[1]={parent_list.shape[1]}"
        )
      parent_bank = int(parent_list[b, parent_slot].item())
      if parent_bank not in bank_to_local:
        raise ValueError(
          f"selected node bank={bank_idx} in batch {b} is missing parent bank={parent_bank}; "
          "top_scores_index must include ancestors for this reference implementation"
        )
      parent = bank_to_local[parent_bank]
      parent_local[b, local] = parent
      depth[b, local] = depth[b, parent] + 1
      children[parent].append((bank_idx, local))

    for parent in range(num_verify_tokens):
      if not children[parent]:
        continue
      children[parent].sort(key=lambda item: item[0])
      ordered = [local for _, local in children[parent]]
      retrive_next_token[b, parent] = ordered[0]
      for idx, child in enumerate(ordered[:-1]):
        retrive_next_sibling[b, child] = ordered[idx + 1]

  positions = (seq_lens.to(torch.int64)[:, None] - 1) + depth
  retrive_index = (
    torch.arange(num_verify_tokens, device=device, dtype=torch.int64).expand(bs, -1).clone()
  )

  qlen_only_mask = _build_tree_mask_from_parent(parent_local)
  mode = _as_mode(tree_mask_mode)
  if mode == TreeMaskMode.QLEN_ONLY:
    tree_mask = qlen_only_mask if tree_mask_buf is None else tree_mask_buf.copy_(qlen_only_mask)
  elif mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
    packed = _pack_mask_bits(qlen_only_mask)
    tree_mask = packed if tree_mask_buf is None else tree_mask_buf.copy_(packed)
  else:
    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() else 0
    full_mask = torch.zeros(
      (bs, num_verify_tokens, max_seq_len + num_verify_tokens),
      dtype=torch.bool,
      device=device,
    )
    for b in range(bs):
      full_mask[b, :, : int(seq_lens[b].item())] = True
      full_mask[b, :, max_seq_len:] = qlen_only_mask[b]
    if seq_lens_sum is not None and seq_lens_sum < int(seq_lens.sum().item()):
      raise ValueError("seq_lens_sum is smaller than sum(seq_lens)")
    tree_mask = full_mask if tree_mask_buf is None else tree_mask_buf.copy_(full_mask)

  return DraftTree(
    draft_tokens=flat_tokens.reshape(-1),
    tree_mask=tree_mask,
    positions=positions,
    retrive_index=retrive_index,
    retrive_next_token=retrive_next_token,
    retrive_next_sibling=retrive_next_sibling,
    parent_local=parent_local,
    depth=depth,
  )


def _expand_batch_view(tensor: torch.Tensor, bs: int) -> torch.Tensor:
  if tensor.ndim == 1:
    return tensor.unsqueeze(0).expand(bs, -1)
  if tensor.ndim == 2 and tensor.shape[0] == bs:
    return tensor
  raise ValueError(
    f"expected tensor with shape [num_nodes] or [bs, num_nodes], got {tuple(tensor.shape)}"
  )


def _sample_from_probs(probs: torch.Tensor, u01: torch.Tensor) -> int:
  probs = probs.clamp_min(0)
  total_mass = probs.sum()
  if not torch.isfinite(total_mass) or total_mass <= 0:
    return int(torch.argmax(probs).item())
  cdf = torch.cumsum(probs, dim=0)
  threshold = u01.clamp(0, 1 - torch.finfo(cdf.dtype).eps) * total_mass
  sampled = torch.searchsorted(cdf, threshold, right=True)
  return int(torch.clamp(sampled, max=probs.shape[0] - 1).item())


def tree_speculative_sampling_target_only(
  candidates: torch.Tensor,
  retrive_index: torch.Tensor,
  retrive_next_token: torch.Tensor,
  retrive_next_sibling: torch.Tensor,
  uniform_samples: torch.Tensor,
  uniform_samples_for_final_sampling: torch.Tensor,
  target_probs: torch.Tensor,
  draft_probs: Optional[torch.Tensor] = None,
  threshold_single: float = 1.0,
  threshold_acc: float = 1.0,
  predicts: Optional[torch.Tensor] = None,
  accept_index: Optional[torch.Tensor] = None,
  accept_token_num: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  Target-only speculative sampling on a first-child / next-sibling tree.

  Returns:
  - predicts: flattened output cache written at `retrive_index`
  - accept_index: per request, positions to gather from `predicts`
  - accept_token_num: accepted draft token count, excluding the final bonus token
  - draft_probs: updated residual-mask buffer with explicitly rejected sibling probs
  """
  if candidates.ndim != 2:
    raise ValueError("candidates must have shape [bs, num_nodes]")
  if target_probs.ndim != 3:
    raise ValueError("target_probs must have shape [bs, num_nodes, vocab]")

  bs, num_nodes = candidates.shape
  if target_probs.shape[:2] != (bs, num_nodes):
    raise ValueError("target_probs leading dims must match candidates")

  retrive_index = _expand_batch_view(retrive_index.to(torch.int64), bs)
  retrive_next_token = _expand_batch_view(retrive_next_token.to(torch.int64), bs)
  retrive_next_sibling = _expand_batch_view(retrive_next_sibling.to(torch.int64), bs)
  uniform_samples = _expand_batch_view(uniform_samples.to(target_probs.dtype), bs)
  if uniform_samples_for_final_sampling.shape != (bs,):
    raise ValueError("uniform_samples_for_final_sampling must have shape [bs]")

  device = candidates.device
  predicts = (
    torch.full((bs, num_nodes), -1, dtype=candidates.dtype, device=device)
    if predicts is None
    else predicts
  )
  accept_index = (
    torch.full((bs, num_nodes), -1, dtype=torch.int64, device=device)
    if accept_index is None
    else accept_index
  )
  accept_token_num = (
    torch.zeros((bs,), dtype=torch.int64, device=device)
    if accept_token_num is None
    else accept_token_num
  )
  draft_probs = torch.zeros_like(target_probs) if draft_probs is None else draft_probs

  for b in range(bs):
    current = 0
    write_pos = int(retrive_index[b, current].item())
    accepted = 0

    while True:
      row = current
      child = int(retrive_next_token[b, current].item())
      coin = float(uniform_samples[b, current].item())
      prob_acc = 0.0
      accepted_child = -1

      while child != -1:
        token_id = int(candidates[b, child].item())
        prob_single = float(target_probs[b, row, token_id].item())
        prob_acc += prob_single
        single_hit = prob_single >= threshold_single
        acc_hit = coin <= (prob_acc / threshold_acc) if threshold_acc > 0 else True
        if single_hit or acc_hit:
          predicts[b, write_pos] = token_id
          accept_index[b, accepted] = write_pos
          accepted += 1
          accepted_child = child
          current = child
          write_pos = int(retrive_index[b, current].item())
          break

        draft_probs[b, row, token_id] = target_probs[b, row, token_id]
        child = int(retrive_next_sibling[b, child].item())

      if accepted_child != -1:
        continue

      residual = (target_probs[b, row] - draft_probs[b, row]).clamp_min(0)
      bonus_token = _sample_from_probs(residual, uniform_samples_for_final_sampling[b])
      predicts[b, write_pos] = bonus_token
      accept_index[b, accepted] = write_pos
      accept_token_num[b] = accepted
      break

  return predicts, accept_index, accept_token_num, draft_probs


__all__ = [
  "DraftTree",
  "TreeMaskMode",
  "build_draft_tree",
  "tree_speculative_sampling_target_only",
]
