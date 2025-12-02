#pragma once
#include <torch/extension.h>

void vector_add_cpu(torch::Tensor a, torch::Tensor b, torch::Tensor out);
void vector_add_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor out);
