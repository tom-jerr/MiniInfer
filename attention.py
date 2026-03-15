import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [..., dim]
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    return x * rms * self.weight


class SwiGLUMLP(nn.Module):
  """
  常见 gated-MLP:
      down( silu(gate_proj(x)) * up_proj(x) )
  """

  def __init__(self, dim: int, hidden_dim: int):
    super().__init__()
    self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
    self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.gate_proj(x)
    sigmoid = 1 / (1 + torch.exp(-x))
    silu = x * sigmoid
    return self.down_proj(silu * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1, x2 = x[..., ::2], x[..., 1::2]
  y = torch.stack((-x2, x1), dim=-1)
  # y: [..., D/2, 2] -> [..., D]
  return y.flatten(-2)


def build_rope_cache(
  max_seq_len: int, dim: int, device: torch.device, rope_theta: float
) -> torch.Tensor:
  # freqs: [N, D/2]
  inv_freqs = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device) / dim))
  seq = torch.arange(max_seq_len, device=device)
  freqs = torch.einsum("i,j->ij", seq, inv_freqs)
  return freqs


def apply_rotary_pos_emb(
  q: torch.Tensor, k: torch.Tensor, freqs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  # q, k: [..., N, D]
  # freqs: [N, D/2]
  def _apply(x: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(freqs).to(dtype=x.dtype)
    sin = torch.sin(freqs).to(dtype=x.dtype)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    while cos.ndim < x_even.ndim:
      cos = cos.unsqueeze(0)
      sin = sin.unsqueeze(0)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)

  return _apply(q), _apply(k)


class MHA(nn.Module):
  def __init__(
    self,
    hidden_dim: int,
    num_heads: int,
    rope_theta: float,
    max_seq_len: int,
    device: torch.device,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads
    self.freqs = build_rope_cache(max_seq_len, self.head_dim, device=device, rope_theta=rope_theta)

    # q, k, v, o
    self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, S, D = x.shape

    q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
    k = self.k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
    v = self.v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
    q, k = apply_rotary_pos_emb(q, k, self.freqs[:S].to(device=x.device))

    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]
    attn_scores = F.softmax(attn, dim=-1)
    out = torch.matmul(attn_scores, v)  # [B, H, N, D/H]
    out = out.transpose(1, 2).contiguous().view(B, S, D)  # [B, N, D]
    return self.o(out)  # [B, N, D]


class GQA(nn.Module):
  def __init__(
    self,
    hidden_dim: int,
    num_heads: int,
    num_kv_head: int,
    rope_theta: float,
    max_seq_len: int,
    device: torch.device,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.num_kv_head = num_kv_head
    self.num_groups = num_heads // num_kv_head
    self.head_dim = hidden_dim // num_heads
    self.freqs = build_rope_cache(max_seq_len, self.head_dim, device=device, rope_theta=rope_theta)

    # q, k, v, o
    self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.k = nn.Linear(hidden_dim, self.num_kv_head * self.head_dim, bias=False)
    self.v = nn.Linear(hidden_dim, self.num_kv_head * self.head_dim, bias=False)
    self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
    B, H_kv, N, D = x.shape
    x = x[:, :, :, None, :].expand(B, H_kv, N, self.num_groups, D)  # [B, H_kv, N, G, D]
    return x.reshape(B, H_kv * self.num_groups, N, D)  # [B, H_q, N, D]

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, S, D = x.shape

    q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
    k = self.k(x).view(B, S, self.num_kv_head, self.head_dim).transpose(1, 2)  # [B, H_kv, N, D/H]
    v = self.v(x).view(B, S, self.num_kv_head, self.head_dim).transpose(1, 2)  # [B, H_kv, N, D/H]
    q, k = apply_rotary_pos_emb(q, k, self.freqs[:S].to(device=x.device))

    k = self._repeat_kv(k)  # [B, H_q, N, D/H]
    v = self._repeat_kv(v)  # [B, H_q, N, D/H]

    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H_q, N, N]
    attn_scores = F.softmax(attn, dim=-1)
    out = torch.matmul(attn_scores, v)  # [B, H_q, N, D/H]
    out = out.transpose(1, 2).contiguous().view(B, S, D)  # [B, N, D]
    return self.o(out)  # [B, N, D]


class MLA(nn.Module):
  def __init__(
    self,
    hidden_dim: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    device: torch.device,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.q_lora_rank = q_lora_rank
    self.kv_lora_rank = kv_lora_rank
    self.qk_nope_head_dim = qk_nope_head_dim
    self.qk_rope_head_dim = qk_rope_head_dim
    self.v_head_dim = v_head_dim
    self.hidden_dim = hidden_dim
    self.scale = (qk_nope_head_dim + qk_rope_head_dim) ** -0.5

    self.freqs = build_rope_cache(
      max_seq_len, qk_rope_head_dim, device=device, rope_theta=rope_theta
    )

    # Q: x -> c_q -> [q_nope, q_rope] per head
    self.q_a = nn.Linear(hidden_dim, q_lora_rank, bias=False)
    self.q_b = nn.Linear(q_lora_rank, num_heads * (qk_nope_head_dim + qk_rope_head_dim), bias=False)

    # KV: x -> [c_kv (latent), k_rope] (k_rope shared across heads)
    self.wkv_a = nn.Linear(hidden_dim, kv_lora_rank + qk_rope_head_dim, bias=False)

    # latent -> per-head [k_nope, v] (used for absorbed weights)
    self.wkv_b = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)

    # Standard output projection parameters. Forward uses absorbed form:
    #   W~o = diag(W_v(1..H)) @ W_o
    self.wo = nn.Linear(num_heads * v_head_dim, hidden_dim, bias=False)

  def _get_absorbed_weight(self):
    """
    wk: [H, d_nope, r]           (W_UK(s)^T)
    w_out_abs: [H, hidden_dim, r]  (W_o,h @ W_v,h)
    """
    H = self.num_heads
    d_nope = self.qk_nope_head_dim
    d_v = self.v_head_dim
    r = self.kv_lora_rank

    w = self.wkv_b.weight.view(H, d_nope + d_v, r)
    wk = w[:, :d_nope, :]  # [H, d_nope, r]
    wv = w[:, d_nope:, :]  # [H, d_v, r]

    wo_h = self.wo.weight.view(self.hidden_dim, H, d_v).permute(1, 0, 2)
    wo_h = wo_h.contiguous()  # [H, hidden_dim, d_v]

    w_out_abs = torch.einsum("hmd,hdr->hmr", wo_h, wv)  # [H, hidden_dim, r]
    return wk, w_out_abs

  def forward(
    self,
    x: torch.Tensor,
  ):
    """
    cache (optional, updated in-place):
      - cache["c_kv"]: [B, S_cache, r]
      - cache["k_rope"]: [B, S_cache, d_rope]  (already RoPE'd)
    """
    B, T, _ = x.shape
    H = self.num_heads
    d_nope = self.qk_nope_head_dim
    d_rope = self.qk_rope_head_dim
    r = self.kv_lora_rank

    # Q
    c_q = self.q_a(x)  # [B,T,q_lora_rank]
    q = self.q_b(c_q).view(B, T, H, d_nope + d_rope).transpose(1, 2)  # [B,H,T,*]
    q_nope, q_rope = q.split([d_nope, d_rope], dim=-1)

    # KV (latent + rope)
    kv_all = self.wkv_a(x)  # [B,T,r+d_rope]
    c_kv, k_rope = kv_all.split([r, d_rope], dim=-1)
    q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, self.freqs)

    S = c_kv.shape[1]

    wk, w_out_abs = self._get_absorbed_weight()  # [H,d_nope,r], [H,hidden,r]
    q_absorb = torch.einsum("bhtd,hdr->bhtr", q_nope, wk)  # [B,H,T,r]

    s_content = torch.einsum("bhtr,bsr->bhts", q_absorb, c_kv)  # [B,H,T,S]
    s_rope = torch.einsum("bhtd,bsd->bhts", q_rope, k_rope)  # [B,H,T,S]
    scores = (s_content + s_rope) * self.scale

    # causal mask in absolute positions
    q_pos = torch.arange(T, device=x.device)
    k_pos = torch.arange(S, device=x.device)
    causal = k_pos[None, None, None, :] > q_pos[None, None, :, None]
    scores = scores.masked_fill(causal, float("-inf"))

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
    """
        torch.einsum("indices1,indices2,...->output_indices", tensor1, tensor2, ...)

        每个字母表示一个维度

        输入里重复出现但输出里没保留的字母：求和

        输出里出现的字母：保留为输出维度

        相同字母表示这些维度要对齐/广播匹配
        """
    u = torch.einsum("bhts,bsr->bhtr", probs, c_kv)  # latent aggregation # [B,H,T,r]
    y = torch.einsum("bhtr,hmr->btm", u, w_out_abs)  # absorbed output # [B, T, D]

    return y
