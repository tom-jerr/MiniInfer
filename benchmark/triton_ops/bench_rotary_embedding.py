import torch
import triton
import triton.testing
import sys
import os

# Add the project root to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.triton.rotary_embedding import (
    apply_rotary_embedding as apply_rotary_embedding_triton,
)


def apply_rotary_embedding_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def test_rope_correctness():
    print("Testing RoPE correctness...")
    torch.manual_seed(0)
    BATCH_SIZE = 1
    SEQ_LEN = 1024
    NUM_HEADS = 32
    HEAD_DIM = 128

    x = torch.randn(
        BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16
    )
    cos = torch.randn(1, SEQ_LEN, HEAD_DIM // 2, device="cuda", dtype=torch.float16)
    sin = torch.randn(1, SEQ_LEN, HEAD_DIM // 2, device="cuda", dtype=torch.float16)

    tri_out = apply_rotary_embedding_triton(x, cos, sin)
    pt_out = apply_rotary_embedding_torch(x, cos, sin, is_neox_style=True)

    if torch.allclose(tri_out, pt_out, atol=1e-2, rtol=1e-2):
        print("✅ RoPE correctness test passed!")
    else:
        print("❌ RoPE correctness test failed!")
        print(f"Max diff: {torch.max(torch.abs(tri_out - pt_out))}")


def run_benchmark_core(BATCH_SIZE, HEAD_DIM, NUM_HEADS, SEQ_LEN, provider):
    x = torch.randn(
        BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16
    )
    cos = torch.randn(1, SEQ_LEN, HEAD_DIM // 2, device="cuda", dtype=torch.float16)
    sin = torch.randn(1, SEQ_LEN, HEAD_DIM // 2, device="cuda", dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_rotary_embedding_torch(x, cos, sin, is_neox_style=True),
            quantiles=quantiles,
        )
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: apply_rotary_embedding_triton(x, cos, sin), quantiles=quantiles
        )
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[1024 * i for i in range(1, 28)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Latency (ms)",
        plot_name="rope-latency",
        args={"BATCH_SIZE": 1, "HEAD_DIM": 128, "NUM_HEADS": 32},
    )
)
def benchmark_latency(BATCH_SIZE, HEAD_DIM, NUM_HEADS, SEQ_LEN, provider):
    ms, min_ms, max_ms = run_benchmark_core(
        BATCH_SIZE, HEAD_DIM, NUM_HEADS, SEQ_LEN, provider
    )
    return ms, max_ms, min_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[1024 * i for i in range(1, 28)],
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Bandwidth (GB/s)",
        plot_name="rope-bandwidth",
        args={"BATCH_SIZE": 1, "HEAD_DIM": 128, "NUM_HEADS": 32},
    )
)
def benchmark_bandwidth(BATCH_SIZE, HEAD_DIM, NUM_HEADS, SEQ_LEN, provider):
    ms, min_ms, max_ms = run_benchmark_core(
        BATCH_SIZE, HEAD_DIM, NUM_HEADS, SEQ_LEN, provider
    )
    # Calculate GB/s
    # x: [B, S, H, D] -> B * S * H * D * 2 bytes
    # cos/sin: [1, S, D/2] -> 1 * S * D/2 * 2 bytes (broadcasted, but read once per S?)
    # Actually, let's approximate total data movement.
    # Read x, Read cos, Read sin, Write output.
    # x size = B * S * H * D * 2
    # output size = B * S * H * D * 2
    # cos size = 1 * S * D/2 * 2
    # sin size = 1 * S * D/2 * 2
    # Total bytes = 2 * x_size + cos_size + sin_size
    x_size = BATCH_SIZE * SEQ_LEN * NUM_HEADS * HEAD_DIM * 2
    cos_size = 1 * SEQ_LEN * (HEAD_DIM // 2) * 2
    sin_size = 1 * SEQ_LEN * (HEAD_DIM // 2) * 2
    total_bytes = 2 * x_size + cos_size + sin_size

    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    test_rope_correctness()
    test_rope_correctness()
    # benchmark_latency.run(
    #     save_path="./benchmark/output", show_plots=False, print_data=True
    # )
    # benchmark_bandwidth.run(
    #     save_path="./benchmark/output", show_plots=False, print_data=True
    # )
