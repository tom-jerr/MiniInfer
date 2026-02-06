import torch
import triton
import triton.language as tl

# rmsnorm


@triton.jit
def rms_norm_kernel(
    x_ptr,
    w_ptr,
    output_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)  # deal with one row
    row_start_ptr = x_ptr + row_idx * stride_x_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_vals = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
    w_vals = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    x_f32_vals = x_vals.to(tl.float32)
    mean_square = tl.sum(x_f32_vals * x_f32_vals, axis=0) / N
    rstd = tl.rsqrt(mean_square + eps)
    output = x_vals * rstd.to(x_vals.dtype) * w_vals
    output_row_start_ptr = output_ptr + row_idx * stride_y_row
    tl.store(output_row_start_ptr + offsets, output, mask=mask)


def rms_norm_forward(x, weight, eps=1e-6):
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1]).contiguous()
    weight = weight.contiguous()
    M, N = x.shape
    output = torch.empty_like(x)

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16

    rms_norm_kernel[grid](
        x,
        weight,
        output,
        x.stride(0),
        output.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return output.view(orig_shape)


@triton.jit
def add_rms_norm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    output_ptr,
    x_new_ptr,
    stride_x_row,
    stride_res_row,
    stride_y_row,
    stride_x_new_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * stride_x_row
    res_start_ptr = res_ptr + row_idx * stride_res_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x_vals = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
    res_vals = tl.load(res_start_ptr + offsets, mask=mask, other=0.0)

    # Fused Add
    acc = x_vals + res_vals

    # Store x_new
    x_new_row_start_ptr = x_new_ptr + row_idx * stride_x_new_row
    tl.store(x_new_row_start_ptr + offsets, acc, mask=mask)

    # RMSNorm on acc
    # acc_f32 = acc.to(tl.float32)
    mean_square = tl.sum(acc * acc, axis=0) / N
    rstd = tl.rsqrt(mean_square + eps)

    w_vals = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    output = acc * rstd * w_vals

    output_row_start_ptr = output_ptr + row_idx * stride_y_row
    tl.store(output_row_start_ptr + offsets, output, mask=mask)


def add_rms_norm_forward(x, residual, weight, eps=1e-5):
    # translate to (Batch*SeqLen, HiddenDim)
    orig_shape = x.shape
    x = x.reshape(-1, orig_shape[-1]).contiguous()
    residual = residual.reshape(-1, orig_shape[-1]).contiguous()
    weight = weight.contiguous()
    M, N = x.shape
    output = torch.empty_like(x)
    x_new = torch.empty_like(x)

    MAX_FUSED_SIZE = 65536
    if N > MAX_FUSED_SIZE:
        raise ValueError(
            f"Fused add_rms_norm not supported for N > {MAX_FUSED_SIZE}, got N={N}"
        )
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    num_warps = 4
    if BLOCK_SIZE >= 2048 and BLOCK_SIZE < 4096:
        num_warps = 8
    elif BLOCK_SIZE >= 4096:
        num_warps = 16

    add_rms_norm_kernel[grid](
        x,
        residual,
        weight,
        output,
        x_new,
        x.stride(0),
        residual.stride(0),
        output.stride(0),
        x_new.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    # Return: (normalized output, new residual)
    return output.view(orig_shape), x_new.view(orig_shape)
