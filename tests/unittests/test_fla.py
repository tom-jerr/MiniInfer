import torch
import pytest
from flash_attn import flash_attn_varlen_func, flash_attn_func
from ..utils import *


@pytest.fixture(scope="module")
def init():
  dtype = torch.float16
  HEAD = 2
  HEAD_DIM = 2
  seqlens = [1, 2, 3, 4]

  query = torch.empty(0, HEAD, HEAD_DIM, dtype=dtype, device="cuda")
  key = torch.empty(0, HEAD, HEAD_DIM, dtype=dtype, device="cuda")
  value = torch.empty(0, HEAD, HEAD_DIM, dtype=dtype, device="cuda")

  querys, keys, values = [], [], []

  for l in seqlens:
    q = torch.rand(l, HEAD, HEAD_DIM, dtype=dtype, device="cuda")
    k = torch.rand(l, HEAD, HEAD_DIM, dtype=dtype, device="cuda")
    v = torch.rand(l, HEAD, HEAD_DIM, dtype=dtype, device="cuda")

    querys.append(q)
    keys.append(k)
    values.append(v)

    query = torch.cat([query, q], dim=0)
    key = torch.cat([key, k], dim=0)
    value = torch.cat([value, v], dim=0)

  return {
    "querys": querys,
    "keys": keys,
    "values": values,
    "query": query,
    "key": key,
    "value": value,
    "seqlens": seqlens,
  }


def test_fla_attn_func(init):
  for q, k, v in zip(init["querys"], init["keys"], init["values"]):
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)

    out = flash_attn_func(q, k, v, causal=True)
    print(out)
