import torch
import torch.nn.functional as F
import triton
import triton.testing
import sys
import os

# Increase dynamo cache limit
import torch._dynamo

torch._dynamo.config.cache_size_limit = 64

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.triton.silu_mul import SiluAndMul


def test_silu_mul_correctness():
  print("Testing SiLU+Mul correctness...")
  torch.manual_seed(0)
  M = 16 * 128  # Batch * SeqLen (small for correctness)
  N = 4096  # Hidden dim

  # Input shape [M, 2N]
  x = torch.randn((M, 2 * N), device="cuda", dtype=torch.float16)

  # PyTorch implementation
  # x is [gate, value]
  gate, value = x.chunk(2, dim=-1)
  pt_out = F.silu(gate) * value

  # Triton implementation
  tri_out = SiluAndMul.silu_and_mul_forward(x)

  # Compare
  if torch.allclose(tri_out, pt_out, atol=1e-2, rtol=1e-2):
    print("✅ SiLU+Mul correctness test passed!")
  else:
    print("❌ SiLU+Mul correctness test failed!")
    print(f"Max diff: {torch.max(torch.abs(tri_out - pt_out))}")


def run_benchmark_core(M, N, provider):
  # Input size is [M, 2N]
  x = torch.randn((M, 2 * N), device="cuda", dtype=torch.float16)

  quantiles = [0.5, 0.2, 0.8]
  if provider == "torch":

    def torch_op(input_tensor):
      gate, value = input_tensor.chunk(2, dim=-1)
      return F.silu(gate) * value

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_op(x), quantiles=quantiles)
  elif provider == "triton":
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: SiluAndMul(x), quantiles=quantiles)

  return ms, min_ms, max_ms


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=["N"],
    x_vals=[1024 * i for i in range(1, 16)],  # N is the output dim. Input dim is 2N.
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    styles=[("blue", "-"), ("green", "-")],
    ylabel="Latency (ms)",
    plot_name="silu-mul-latency",
    args={"M": 4096},
  )
)
def benchmark_latency(M, N, provider):
  ms, min_ms, max_ms = run_benchmark_core(M, N, provider)
  return ms, max_ms, min_ms


@triton.testing.perf_report(
  triton.testing.Benchmark(
    x_names=["N"],
    x_vals=[1024 * i for i in range(1, 16)],
    line_arg="provider",
    line_vals=["triton", "torch"],
    line_names=["Triton", "Torch"],
    styles=[("blue", "-"), ("green", "-")],
    ylabel="Bandwidth (GB/s)",
    plot_name="silu-mul-bandwidth",
    args={"M": 4096},
  )
)
def benchmark_bandwidth(M, N, provider):
  ms, min_ms, max_ms = run_benchmark_core(M, N, provider)
  # Calculate GB/s
  # Total IO: Read x (M * 2N), Write y (M * N)
  # Element size = 2 bytes (float16)
  total_bytes = (M * 2 * N + M * N) * 2

  gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
  return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
  test_silu_mul_correctness()
  benchmark_latency.run(save_path="./benchmark/output", show_plots=False, print_data=True)
  benchmark_bandwidth.run(save_path="./benchmark/output", show_plots=False, print_data=True)
