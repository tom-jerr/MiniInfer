import math

import torch
from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch.utils.flop_counter import FlopCounterMode


def safe_self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, sm_scale):
  bs, seqlen, numhead, headdim = q.shape
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)

  qk = q @ k.transpose(2, 3)
  qk *= sm_scale

  if is_causal:
    mask = torch.tril(torch.ones(seqlen, seqlen, device=q.device))
    qk = qk.masked_fill(mask == 0, float("-inf"))

  # optional: use higher precision to do softmax
  qk = qk.float()

  #
  # safe softmax
  #
  row_max = qk.max(dim=-1, keepdim=True).values
  # safe score
  score = torch.exp(qk - row_max)
  score_sum = score.sum(dim=-1, keepdim=True)
  s = score / score_sum

  # #
  # # naive softmax
  # #
  # s = torch.softmax(qk, dim=-1)

  o = s.to(q.dtype) @ v
  o = o.transpose(1, 2)

  return o


def flash_attention_v1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, sm_scale):
  """
  flash attention: Three Easy Pieces
      1. block tiling -> low intermediate data
      2. two gemm fused: gemm-I (q@k), gemm-II (s@v)
      3. online safe softmax: math equal fix up for block tiling

  algo(v2):
  parallel for {bsz, numhead, seqlen}
      for block_n in kvlen:
          gemm-I: s[1, block_n] = q[1, headdim] @ k[block_n, headdim].T
              1 for 1 of seqlen (typically block_m)
          online safe softmax
              safe softmax: exp(x - max)
              online safe softmax: exp(x - local_m) * rescale = global_sm
                  exp(x - local_m) * rescale = e^{x - local_m} * e^{local_m - new_m}
                                             = e^{x - local_m + local_m - new_m}
                                             = e^{x - new_m}
          gemm-II: o += s[1, block_n] @ v[block_n, headdim]
  """
  # NOTE: tiling size (terms):
  # q_tile = [block_m, headdim]
  # k_tile = [block_n, headdim]
  # v_tile = [block_n, headdim]
  # o_tile = q_tile = [block_m, headdim]
  block_m = 32
  block_n = 64

  bs, seqlen, numhead, headdim = q.shape
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)

  assert seqlen % block_m == 0 and seqlen % block_n == 0, "Simple for now."

  o = torch.empty_like(q)

  # parallel for in gpu
  for bid in range(bs):
    # parallel for in gpu
    for hid in range(numhead):
      ######################
      #   Global Memory
      ######################

      # NOTE:
      # FA1: overview
      # https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdMfo8veQRzPgt-PsLup9ttAZmMdufdV3N3Q&s

      # NOTE: share via global memory
      # need to gmem -> smem -> reg
      o_all = o.view(bs, numhead, seqlen // block_m, block_m, headdim)[bid, hid, :, :, :]
      l_all = torch.zeros((seqlen // block_m, block_m, 1))
      m_all = torch.ones((seqlen // block_m, block_m, 1)) * -torch.inf

      # parallel for in gpu
      for j_tile, kv_start in enumerate(range(0, seqlen, block_n)):
        ######################
        #   Shared Memory
        ######################
        k_tile = k[bid, hid, kv_start : kv_start + block_n, :]
        v_tile = v[bid, hid, kv_start : kv_start + block_n, :]

        for i_tile, q_start in enumerate(range(0, seqlen, block_m)):
          q_tile = q[bid, hid, q_start : q_start + block_m, :]

          # since kv iter is outter loop. max, sum, out must shared via global memory
          o_i = o_all[i_tile]
          l_i = l_all[i_tile]
          m_i = m_all[i_tile]

          # Skip tiles that are fully masked by the causal constraint.
          if is_causal and kv_start >= q_start + block_m:
            continue

          qk = q_tile @ k_tile.T
          qk = qk.float()

          if is_causal:
            row_indices = torch.arange(block_m, device=q.device)[:, None]
            col_indices = torch.arange(block_n, device=q.device)[None, :]
            absolute_pos_q = q_start + row_indices
            absolute_pos_k = kv_start + col_indices
            causal_mask = absolute_pos_k > absolute_pos_q
            qk = qk.masked_fill(causal_mask, float("-inf"))

          qk = qk * sm_scale
          m_ij = qk.max(dim=-1, keepdim=True).values
          p_ij = torch.exp(qk - m_ij)
          l_ij = p_ij.sum(dim=-1, keepdim=True)

          m_new = torch.maximum(m_ij, m_i)
          # rescale
          l_new = l_i * torch.exp(m_i - m_new) + l_ij * torch.exp(m_ij - m_new)
          o_new = l_i * o_i * torch.exp(m_i - m_new) + p_ij.to(q.dtype) @ v_tile * torch.exp(
            m_ij - m_new
          )

          m_i.copy_(m_new)
          l_i.copy_(l_new)
          o_i.copy_(o_new / l_new)

  o = o.transpose(1, 2)
  return o


def flash_attention_v2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal, sm_scale):
  """
  NOTE: what's different vs fa v1
      switch inner and outter loop
      v1:
          outter loop: iter over kv
          inner loop: iter over q and o
      v2:
          outter loop: iter over q and o
          inner loop: iter over kv
      so that
          1. less output tensor IO
          2. less output tensor rescale
          3. combine output tensor at the end(epilogue)

      you can simply checkout the 66ring/ans branch for a quick look.
  """
  # NOTE: tiling size (terms):
  # q_tile = [block_m, headdim]
  # k_tile = [block_n, headdim]
  # v_tile = [block_n, headdim]
  # o_tile = q_tile = [block_m, headdim]
  block_m = 32
  block_n = 64

  bs, seqlen, numhead, headdim = q.shape
  q = q.transpose(1, 2)
  k = k.transpose(1, 2)
  v = v.transpose(1, 2)

  assert seqlen % block_m == 0 and seqlen % block_n == 0, "Simple for now."

  o = torch.empty_like(q)

  # parallel for in gpu
  for bid in range(bs):
    # parallel for in gpu
    for hid in range(numhead):
      o_all = o.view(bs, numhead, seqlen // block_m, block_m, headdim)[bid, hid, :, :, :]
      ######################
      #   Global Memory
      ######################

      # parallel for in gpu
      for i_tile, q_start in enumerate(range(0, seqlen, block_m)):
        ######################
        #   Shared Memory
        ######################

        # >>>
        # >>> YOUR CORE HERE.
        # >>>
        q_tile = q[bid, hid, q_start : q_start + block_m, :]
        # a reference here, not materialize. no ldg usage.
        o_tile = o_all[i_tile]

        o_i = torch.zeros_like(o_tile)
        l_i = torch.zeros((block_m, 1))
        m_i = torch.ones((block_m, 1)) * -torch.inf

        for j_tile, kv_start in enumerate(range(0, seqlen, block_n)):
          # >>>
          # >>> YOUR CORE HERE.
          # >>>
          k_tile = k[bid, hid, kv_start : kv_start + block_n, :]
          v_tile = v[bid, hid, kv_start : kv_start + block_n, :]

          # Skip tiles that are fully masked by the causal constraint.
          if is_causal and kv_start >= q_start + block_m:
            continue

          qk = q_tile @ k_tile.T
          qk = qk.float()

          if is_causal:
            row_indices = torch.arange(block_m, device=q.device)[:, None]
            col_indices = torch.arange(block_n, device=q.device)[None, :]
            absolute_pos_q = q_start + row_indices
            absolute_pos_k = kv_start + col_indices
            causal_mask = absolute_pos_k > absolute_pos_q
            qk = qk.masked_fill(causal_mask, float("-inf"))

          qk = qk * sm_scale
          m_ij = qk.max(dim=-1, keepdim=True).values
          p_ij = torch.exp(qk - m_ij)
          l_ij = p_ij.sum(dim=-1, keepdim=True)

          m_new = torch.maximum(m_ij, m_i)
          l_new = l_i * torch.exp(m_i - m_new) + l_ij * torch.exp(m_ij - m_new)
          o_new = o_i * torch.exp(m_i - m_new) + p_ij.to(q.dtype) @ v_tile * torch.exp(m_ij - m_new)

          m_i.copy_(m_new)
          l_i.copy_(l_new)
          o_i.copy_(o_new)

        o_tile.copy_(o_i / l_i)

  o = o.transpose(1, 2)
  return o


def get_tensors(BS, SEQLEN, HEAD, DIM):
  q = torch.randn((BS, SEQLEN, HEAD, DIM)).normal_(mean=0.0, std=0.5)
  k = torch.randn((BS, SEQLEN, HEAD, DIM)).normal_(mean=0.0, std=0.5)
  v = torch.randn((BS, SEQLEN, HEAD, DIM)).normal_(mean=0.0, std=0.5)

  return q, k, v


@torch.no_grad()
def main():
  torch.manual_seed(13)
  torch.set_default_device("cuda")
  torch.set_default_dtype(torch.bfloat16)

  BS, SEQLEN, HEAD, DIM = 3, 512, 8, 128
  q, k, v = get_tensors(BS, SEQLEN, HEAD, DIM)
  scale = 1 / math.sqrt(DIM)
  is_causal = True

  counter = FlopCounterMode(display=False)

  with counter:
    o = safe_self_attention(q, k, v, is_causal=is_causal, sm_scale=scale)
  print(f"torch self attention flops: {counter.get_total_flops()}")

  with counter:
    sdpa_o = sdpa(
      q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=is_causal, scale=scale
    ).transpose(1, 2)
  print(f"sdpa flops: {counter.get_total_flops()}")

  fa_o = flash_attn_func(q, k, v, causal=is_causal, softmax_scale=scale)
  fa_v1 = flash_attention_v1(q, k, v, is_causal=is_causal, sm_scale=scale)

  fa_v2 = flash_attention_v2(q, k, v, is_causal=is_causal, sm_scale=scale)

  torch.testing.assert_close(fa_o, o, atol=1e-2, rtol=1e-2)
  torch.testing.assert_close(sdpa_o, o, atol=1e-2, rtol=1e-2)
  torch.testing.assert_close(fa_v1, o, atol=1e-2, rtol=1e-2)
  torch.testing.assert_close(fa_v2, o, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  main()
