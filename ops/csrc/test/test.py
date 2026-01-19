import torch
from miniinfer_ext import vector_add

a = torch.ones(3, dtype=torch.float32)
b = torch.ones(3, dtype=torch.float32)
out = vector_add(a, b)
print("a + b =", out)
