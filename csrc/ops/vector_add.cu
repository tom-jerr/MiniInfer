#include <torch/extension.h>

#include "vector_add.h"

__global__ void vector_add_kernel(float* a, float* b, float* out, int64_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  int64_t n = a.numel();
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                         out.data_ptr<float>(), n);
}

// CPU Implement

void vector_add_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  auto a_data = a.data_ptr<float>();
  auto b_data = b.data_ptr<float>();
  auto out_data = out.data_ptr<float>();

  int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    out_data[i] = a_data[i] + b_data[i];
  }
}
