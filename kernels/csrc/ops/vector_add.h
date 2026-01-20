#pragma once

#include <torch/torch.h>

// CUDA implementation
void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out);

// CPU implementation
void vector_add_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor out);
