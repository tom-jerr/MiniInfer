#include <torch/extension.h>

#include "vector_add.h"

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
  // 检查输入
  TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
  TORCH_CHECK(a.is_cuda() == b.is_cuda(), "Both tensors must be on the same device");

  auto result = torch::zeros_like(a);
  if (a.is_cuda()) {
    vector_add_cuda(a, b, result);
  } else {
    vector_add_cpu(a, b, result);
  }
  return result;
}

// pybind11 注册函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vector_add", &vector_add, "Vector addition (CPU + CUDA)");
}
