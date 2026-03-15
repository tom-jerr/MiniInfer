import math

import torch
import triton
import triton.language as tl


SUPPORTED_HEAD_DIMS = {16, 32, 64, 128}


@triton.jit
def _splitk_stage1_kernel(
  Q,
  K,
  V,
  sm_scale,
  M_BUF,
  L_BUF,
  ACC_BUF,
  stride_q_bh,
  stride_q_d,
  stride_k_bh,
  stride_k_n,
  stride_k_d,
  stride_v_bh,
  stride_v_n,
  stride_v_d,
  stride_ml_bh,
  stride_ml_split,
  stride_acc_bh,
  stride_acc_split,
  stride_acc_d,
  KV_LEN,
  NUM_SPLITS,
  HEAD_DIM,
  BLOCK_N: tl.constexpr,
  BLOCK_DMODEL: tl.constexpr,
):
  split_id = tl.program_id(0)
  off_bh = tl.program_id(1)

  offs_d = tl.arange(0, BLOCK_DMODEL)
  d_mask = offs_d < HEAD_DIM

  q_ptrs = Q + off_bh * stride_q_bh + offs_d * stride_q_d
  q = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
  q = q * (sm_scale * 1.44269504)

  split_size = tl.cdiv(KV_LEN, NUM_SPLITS)
  split_start = split_id * split_size
  split_end = tl.minimum(split_start + split_size, KV_LEN)

  m_i = -float("inf")
  l_i = 0.0
  o_i = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

  for start_n in range(split_start, split_end, BLOCK_N):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    n_mask = offs_n < split_end

    # [BLOCK_DMODEL, BLOCK_N]
    k_ptrs = K + off_bh * stride_k_bh + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d
    # [BLOCK_N, BLOCK_DMODEL]
    v_ptrs = V + off_bh * stride_v_bh + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d

    # broadcast [BLOCK_DMODEL] to [BLOCK_DMODEL, BLOCK_N] for k, and [BLOCK_N, BLOCK_DMODEL] for v
    kv_mask = d_mask[:, None] & n_mask[None, :]
    vd_mask = n_mask[:, None] & d_mask[None, :]

    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=vd_mask, other=0.0)

    qk = tl.sum(k * q[:, None], axis=0)
    qk = tl.where(n_mask, qk, -float("inf"))

    m_ij = tl.max(qk, axis=0)
    p = tl.math.exp2(qk - m_ij)
    p = tl.where(n_mask, p, 0.0)
    l_ij = tl.sum(p, axis=0)
    o_ij = tl.sum(v * p[:, None], axis=0)

    m_new = tl.maximum(m_i, m_ij)
    alpha = tl.math.exp2(m_i - m_new)
    beta = tl.math.exp2(m_ij - m_new)
    l_i = l_i * alpha + l_ij * beta
    o_i = o_i * alpha + o_ij * beta
    m_i = m_new

  m_ptr = M_BUF + off_bh * stride_ml_bh + split_id * stride_ml_split
  l_ptr = L_BUF + off_bh * stride_ml_bh + split_id * stride_ml_split
  o_ptrs = ACC_BUF + off_bh * stride_acc_bh + split_id * stride_acc_split + offs_d * stride_acc_d

  tl.store(m_ptr, m_i)
  tl.store(l_ptr, l_i)
  tl.store(o_ptrs, o_i, mask=d_mask)


@triton.jit
def _splitk_reduce_kernel(
  M_BUF,
  L_BUF,
  ACC_BUF,
  O,
  stride_ml_bh,
  stride_ml_split,
  stride_acc_bh,
  stride_acc_split,
  stride_acc_d,
  stride_o_bh,
  stride_o_d,
  NUM_SPLITS,
  HEAD_DIM,
  BLOCK_SPLIT_K: tl.constexpr,
  BLOCK_DMODEL: tl.constexpr,
):
  off_bh = tl.program_id(0)

  offs_split = tl.arange(0, BLOCK_SPLIT_K)
  split_mask = offs_split < NUM_SPLITS
  offs_d = tl.arange(0, BLOCK_DMODEL)
  d_mask = offs_d < HEAD_DIM

  m_ptrs = M_BUF + off_bh * stride_ml_bh + offs_split * stride_ml_split
  l_ptrs = L_BUF + off_bh * stride_ml_bh + offs_split * stride_ml_split

  m = tl.load(m_ptrs, mask=split_mask, other=-float("inf"))
  l = tl.load(l_ptrs, mask=split_mask, other=0.0)

  m_final = tl.max(m, axis=0)
  alpha: tl.tensor = tl.math.exp2(m - m_final)
  alpha = tl.where(split_mask, alpha, 0.0)
  l_final = tl.sum(l * alpha, axis=0)

  # [B_SPLIT_K, BLOCK_DMODEL]
  acc_ptrs = (
    ACC_BUF
    + off_bh * stride_acc_bh
    + offs_split[:, None] * stride_acc_split
    + offs_d[None, :] * stride_acc_d
  )
  acc_mask = split_mask[:, None] & d_mask[None, :]
  acc_split = tl.load(acc_ptrs, mask=acc_mask, other=0.0)
  # 之所以放在这里乘，是因为只有 reduce 阶段才知道全局 m_final
  acc = tl.sum(acc_split * alpha[:, None], axis=0)

  out = acc / l_final
  o_ptrs = O + off_bh * stride_o_bh + offs_d * stride_o_d
  tl.store(o_ptrs, out, mask=d_mask)


def flash_decode_reference(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  sm_scale: float | None = None,
) -> torch.Tensor:
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError("Expected q, k, v to have shape [batch, heads, seqlen, dim].")
  if q.shape[2] != 1:
    raise ValueError("flash decoding reference expects query length == 1.")
  if k.shape != v.shape:
    raise ValueError("k and v must have the same shape.")
  if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
    raise ValueError("q/k/v batch, head, and dim must match.")

  if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(q.shape[-1])

  scores = torch.matmul(q.float(), k.transpose(-1, -2).float()) * sm_scale
  probs = torch.softmax(scores, dim=-1)
  return torch.matmul(probs.to(v.dtype), v)


def flash_decode_splitk(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  sm_scale: float | None = None,
  num_splits: int | None = None,
  block_n: int = 64,
) -> torch.Tensor:
  if not q.is_cuda or not k.is_cuda or not v.is_cuda:
    raise ValueError("flash_decode_splitk requires CUDA tensors.")
  if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
    raise ValueError("Expected q, k, v to have shape [batch, heads, seqlen, dim].")
  if q.shape[2] != 1:
    raise ValueError("flash decoding expects query length == 1.")
  if k.shape != v.shape:
    raise ValueError("k and v must have the same shape.")
  if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
    raise ValueError("q/k/v batch, head, and dim must match.")

  batch, heads, _, dim = q.shape
  kv_len = k.shape[2]
  if kv_len == 0:
    raise ValueError("kv_len must be positive.")
  if dim not in SUPPORTED_HEAD_DIMS:
    raise ValueError(f"Unsupported head_dim {dim}. Expected one of {sorted(SUPPORTED_HEAD_DIMS)}.")

  if sm_scale is None:
    sm_scale = 1.0 / math.sqrt(dim)
  if num_splits is None:
    num_splits = min(16, max(1, triton.cdiv(kv_len, 256)))
  if num_splits <= 0:
    raise ValueError("num_splits must be positive.")

  bh = batch * heads
  q_flat = q[:, :, 0, :].contiguous().view(bh, dim)
  k_flat = k.contiguous().view(bh, kv_len, dim)
  v_flat = v.contiguous().view(bh, kv_len, dim)

  m_buf = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)
  l_buf = torch.empty((bh, num_splits), device=q.device, dtype=torch.float32)
  acc_buf = torch.empty((bh, num_splits, dim), device=q.device, dtype=torch.float32)
  out = torch.empty((bh, dim), device=q.device, dtype=torch.float32)

  grid_stage1 = (num_splits, bh)
  num_warps = 4 if dim <= 64 else 8
  _splitk_stage1_kernel[grid_stage1](
    q_flat,
    k_flat,
    v_flat,
    sm_scale,
    m_buf,
    l_buf,
    acc_buf,
    q_flat.stride(0),
    q_flat.stride(1),
    k_flat.stride(0),
    k_flat.stride(1),
    k_flat.stride(2),
    v_flat.stride(0),
    v_flat.stride(1),
    v_flat.stride(2),
    m_buf.stride(0),
    m_buf.stride(1),
    acc_buf.stride(0),
    acc_buf.stride(1),
    acc_buf.stride(2),
    kv_len,
    num_splits,
    dim,
    BLOCK_N=block_n,
    BLOCK_DMODEL=dim,
    num_warps=num_warps,
    num_stages=2,
  )

  grid_reduce = (bh,)
  _splitk_reduce_kernel[grid_reduce](
    m_buf,
    l_buf,
    acc_buf,
    out,
    m_buf.stride(0),
    m_buf.stride(1),
    acc_buf.stride(0),
    acc_buf.stride(1),
    acc_buf.stride(2),
    out.stride(0),
    out.stride(1),
    num_splits,
    dim,
    BLOCK_SPLIT_K=triton.next_power_of_2(num_splits),
    BLOCK_DMODEL=dim,
    num_warps=num_warps,
    num_stages=2,
  )

  return out.to(q.dtype).view(batch, heads, 1, dim)


if __name__ == "__main__":
  torch.manual_seed(0)

  batch, heads, dim = 2, 4, 64
  num_splits = 4
  kv_len = 256
  dtype = torch.float16
  q = torch.randn((batch, heads, 1, dim), device="cuda", dtype=dtype)
  k = torch.randn((batch, heads, kv_len, dim), device="cuda", dtype=dtype)
  v = torch.randn((batch, heads, kv_len, dim), device="cuda", dtype=dtype)
  sm_scale = 1.0 / math.sqrt(dim)

  out = flash_decode_splitk(q, k, v, sm_scale=sm_scale, num_splits=num_splits)
  ref = flash_decode_reference(q, k, v, sm_scale=sm_scale)

  torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
