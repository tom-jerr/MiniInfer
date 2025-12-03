from typing import Optional

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from torch import Tensor

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 128}, num_warps=4),
        triton.Config({"BLOCK_HS_HALF": 256}, num_warps=8),
    ],
    key=["head_size", "interleaved"],
)
@triton.jit
def _rotary_embedding_kernel(
  output_ptr,
  x_ptr,
  cos_ptr,
  sin_ptr,
  num_head,
  head_size,
  num_tokens,
  stride_x_row,
  stride_cos_batch,
  stride_cos_seq,
  stride_sin_batch,
  stride_sin_seq,
  interleaved: tl.constexpr,
  BLOCK_HS_HALF: tl.constexpr,
):
  row_idx = tl.program_id(0) # 展平后的 (Batch, Seq, Head) 的线性索引
  
  # row_idx = b * (num_tokens * num_head) + s * num_head + h
  batch_idx = row_idx // (num_tokens * num_head)
  rem = row_idx % (num_tokens * num_head)
  token_idx = rem // num_head

  # locate current head
  x_row_ptr = x_ptr + row_idx * stride_x_row 
  output_row_ptr = output_ptr + row_idx * stride_x_row
  # different head use the same cos/sin values at the same token position
  cos_row_ptr = cos_ptr + batch_idx * stride_cos_batch + token_idx * stride_cos_seq
  sin_row_ptr = sin_ptr + batch_idx * stride_sin_batch + token_idx * stride_sin_seq

  head_size_half = head_size // 2

  # 一次处理 BLOCK_HS_HALF 个配对元素
  for block_start in range(0, head_size_half, BLOCK_HS_HALF):
    offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
    mask = offsets_half < head_size_half

    cos_vals = tl.load(cos_row_ptr + offsets_half, mask=mask, other=0.0)
    sin_vals = tl.load(sin_row_ptr + offsets_half, mask=mask, other=0.0)

    if interleaved:
        offsets_x1 = offsets_half * 2
        offsets_x2 = offsets_half * 2 + 1
    else:
      offsets_x1 = offsets_half
      offsets_x2 = offsets_half + head_size_half

    x1_vals = tl.load(x_row_ptr + offsets_x1, mask=mask, other=0.0)
    x2_vals = tl.load(x_row_ptr + offsets_x2, mask=mask, other=0.0)

    x1_fp32 = x1_vals.to(tl.float32)
    x2_fp32 = x2_vals.to(tl.float32)
    cos_fp32 = cos_vals.to(tl.float32)
    sin_fp32 = sin_vals.to(tl.float32)
    
    o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
    o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

    tl.store(output_row_ptr + offsets_x1, o1_vals.to(x1_vals.dtype), mask=mask)
    tl.store(output_row_ptr + offsets_x2, o2_vals.to(x2_vals.dtype), mask=mask)

def apply_rotary_embedding(
      x:torch.Tensor,
      cos:torch.Tensor,
      sin:torch.Tensor,
      interleaved:bool = False,
)->torch.Tensor:
    # Triton kernels only work on CUDA devices
    if not x.is_cuda:
        # Fallback to PyTorch implementation for CPU
        # This is a simplified implementation for CPU fallback
        
        # Ensure cos/sin have the same number of dimensions as x for broadcasting
        # x: [B, S, H, D]
        # cos: [1, S, D/2] -> need [1, S, 1, D/2]
        if cos.dim() == 3:
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)

        if interleaved:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos
            output = torch.empty_like(x)
            output[..., ::2] = o1
            output[..., 1::2] = o2
            return output
        else:
            head_size = x.shape[-1]
            x1 = x[..., :head_size//2]
            x2 = x[..., head_size//2:]
            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos
            return torch.cat([o1, o2], dim=-1)

    output = torch.empty_like(x)

    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x = x.contiguous()
    x_reshaped = x.view(-1, head_size)
    output_reshaped = output.view(-1, head_size)

    # num_tokens per head, 1 token per block
    grid = (bsz * num_tokens * num_heads,)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos[..., :head_size//2].contiguous()
        sin = sin[...,:head_size//2].contiguous()

    stride_cos_batch = cos.stride(0)
    stride_cos_seq = cos.stride(1)
    stride_sin_batch = sin.stride(0)
    stride_sin_seq = sin.stride(1)

    if cos.shape[0] == 1 and bsz > 1:
        stride_cos_batch = 0
        stride_sin_batch = 0

    _rotary_embedding_kernel[grid](
        output_reshaped,
        x_reshaped,
        cos,
        sin,
        num_heads,
        head_size,
        num_tokens,
        x_reshaped.stride(0),
        stride_cos_batch,
        stride_cos_seq,
        stride_sin_batch,
        stride_sin_seq,
        interleaved,
    )

    return output
