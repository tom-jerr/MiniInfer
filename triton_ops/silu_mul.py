import torch
import triton
import triton.language as tl

@triton.jit
def silu_and_mul_kernel(
    input_ptr,
    output_ptr,
    stride_input_m,
    stride_input_n,
    stride_output_m,
    stride_output_n,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    input_row_start = input_ptr + pid_m * stride_input_m
    
    # Load part1 (Gate)
    part1_ptr = input_row_start + offs_n * stride_input_n
    mask = offs_n < N
    gate = tl.load(part1_ptr, mask=mask, other=0.0)
    
    # Load part2 (up)
    # Assuming part1 and part2 are concatenated along the last dimension
    # part1 is [0, N), part2 is [N, 2N)
    part2_ptr = input_row_start + (offs_n + N) * stride_input_n
    up = tl.load(part2_ptr, mask=mask, other=0.0)
    
    # Compute SiLU(gate) * up
    # SiLU(x) = x * sigmoid(x)
    # Triton's sigmoid currently requires fp32
    gate_f32 = gate.to(tl.float32)
    silu_gate = gate_f32 * tl.sigmoid(gate_f32)
    output = silu_gate.to(gate.dtype) * up
    
    # Store result
    output_row_start = output_ptr + pid_m * stride_output_m
    output_ptr_curr = output_row_start + offs_n * stride_output_n
    tl.store(output_ptr_curr, output, mask=mask)

def silu_and_mul_forward(x):
    """
    Computes SiLU(x_gate) * x_up
    Input x is assumed to be of shape [M, 2*N], where:
    x_gate = x[:, N:]
    x_up = x[:, :N]
    """
    orginal_shape = x.shape
    x = x.reshape(-1, orginal_shape[-1]).contiguous()
    M, N2 = x.shape
    assert N2 % 2 == 0, "Input dimension must be even"
    N = N2 // 2
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_SIZE']))
    
    silu_and_mul_kernel[grid](
        x, output,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.reshape(*orginal_shape[:-1], N)
