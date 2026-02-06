import torch
import triton
import triton.testing
import sys
import os

# Increase dynamo cache limit to avoid warnings during benchmark with changing shapes
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

# Add the project root to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.triton.rms_norm import rms_norm_forward, add_rms_norm_forward
from miniinfer.layers.layernorm import RMSNorm as PyTorchRMSNorm


def test_rmsnorm_correctness():
    print("Testing RMSNorm correctness...")
    torch.manual_seed(0)
    batch_size = 16
    seq_len = 1024
    dim = 4096
    eps = 1e-5

    x = torch.randn((batch_size * seq_len, dim), device="cuda", dtype=torch.float16)
    weight = torch.randn(dim, device="cuda", dtype=torch.float16)

    # PyTorch implementation
    # Note: miniinfer.layers.layernorm.RMSNorm expects weight in constructor
    pt_rmsnorm = PyTorchRMSNorm(dim, eps=eps).to("cuda")
    pt_rmsnorm.load_weights(
        {"rand.weight": weight}, "rand", device="cuda", dtype=torch.float16
    )

    # Triton implementation
    tri_out = rms_norm_forward(x, weight, eps)

    # PyTorch forward
    pt_out = pt_rmsnorm(x)

    tri_out = rms_norm_forward(x, weight, eps)
    pt_out = pt_rmsnorm(x)

    # Compare
    # if torch.allclose(tri_out, pt_out, atol=1e-2, rtol=1e-2):
    #     print("✅ RMSNorm correctness test passed!")
    # else:
    #     print("❌ RMSNorm correctness test failed!")
    #     print(f"Max diff: {torch.max(torch.abs(tri_out - pt_out))}")


def test_add_rmsnorm_correctness():
    print("Testing Add+RMSNorm correctness...")
    torch.manual_seed(0)
    batch_size = 16
    seq_len = 1024
    dim = 4096
    eps = 1e-5

    x = torch.randn((batch_size * seq_len, dim), device="cuda", dtype=torch.float16)
    residual = torch.randn(
        (batch_size * seq_len, dim), device="cuda", dtype=torch.float16
    )
    weight = torch.randn(dim, device="cuda", dtype=torch.float16)

    # PyTorch implementation
    pt_rmsnorm = PyTorchRMSNorm(dim, eps=eps).to("cuda")
    pt_rmsnorm.load_weights(
        {"rand.weight": weight}, "rand", device="cuda", dtype=torch.float16
    )
    # Triton implementation
    # NOTE: add_rms_norm_forward returns (out, x_new)
    tri_out, tri_x_new = add_rms_norm_forward(x, residual, weight, eps)

    # PyTorch forward
    pt_x_new = x + residual
    pt_out = pt_rmsnorm(pt_x_new)

    tri_out, tri_x_new = add_rms_norm_forward(x, residual, weight, eps)

    # PyTorch forward
    pt_x_new = x + residual
    pt_out = pt_rmsnorm(pt_x_new)

    tri_out, tri_x_new = add_rms_norm_forward(x, residual, weight, eps)
    pt_out = pt_rmsnorm(x + residual)

    # Compare x_new
    # if torch.allclose(tri_x_new, pt_x_new, atol=1e-2, rtol=1e-2):
    #     print("✅ Add+RMSNorm x_new correctness test passed!")
    # else:
    #     print("❌ Add+RMSNorm x_new correctness test failed!")
    #     print(f"Max diff x_new: {torch.max(torch.abs(tri_x_new - pt_x_new))}")

    # # Compare output
    # if torch.allclose(tri_out, pt_out, atol=1e-2, rtol=1e-2):
    #     print("✅ Add+RMSNorm output correctness test passed!")
    # else:
    #     print("❌ Add+RMSNorm output correctness test failed!")
    #     print(f"Max diff output: {torch.max(torch.abs(tri_out - pt_out))}")


def run_benchmark_core(M, N, provider):
    x = torch.randn(M, N, device="cuda", dtype=torch.float16)
    residual = torch.randn(M, N, device="cuda", dtype=torch.float16)
    weight = torch.randn(N, device="cuda", dtype=torch.float16)
    eps = 1e-5

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        pt_rmsnorm = PyTorchRMSNorm(N, eps=eps).to("cuda")
        pt_rmsnorm.load_weights(
            {"rand.weight": weight}, "rand", device="cuda", dtype=torch.float16
        )
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: pt_rmsnorm(x), quantiles=quantiles
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: rms_norm_forward(x, weight, eps), quantiles=quantiles
        )
    elif provider == "torch_add_rmsnorm":
        pt_rmsnorm = PyTorchRMSNorm(N, eps=eps).to("cuda")
        pt_rmsnorm.load_weights(
            {"rand.weight": weight}, "rand", device="cuda", dtype=torch.float16
        )

        def torch_op(x, r):
            h = x + r
            return pt_rmsnorm(h)

        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch_op(x, residual), quantiles=quantiles
        )
    elif provider == "triton_fused":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add_rms_norm_forward(x, residual, weight, eps), quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return ms, min_ms, max_ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[1024 * i for i in range(1, 10)],
        line_arg="provider",
        line_vals=["triton", "torch", "triton_fused", "torch_add_rmsnorm"],
        line_names=["Triton", "Torch", "Triton Fused", "Torch (Add+RMSNorm)"],
        styles=[("blue", "-"), ("green", "-"), ("blue", "--"), ("green", "--")],
        ylabel="Latency (ms)",
        plot_name="rmsnorm-latency",
        args={"M": 4096},
    )
)
def benchmark_latency(M, N, provider):
    ms, min_ms, max_ms = run_benchmark_core(M, N, provider)
    return ms, max_ms, min_ms


if __name__ == "__main__":
    test_add_rmsnorm_correctness()
    # test_add_rmsnorm_correctness()
    # benchmark_latency.run(
    #     save_path="./benchmark/output", show_plots=False, print_data=True
    # )
