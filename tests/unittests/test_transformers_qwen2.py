import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
  Qwen2Attention,
  Qwen2RotaryEmbedding,
)


def test_qwen2_gqa_attention(
  mask: str | None,
  dtype: torch.dtype,
  device: torch.device,
):
  dev = torch.device(device)

  batch_size = 1
  seq_len = 4
  hidden_size = 32
  num_heads = 4
  num_kv_heads = 2
  max_seq_len = 64
  theta = 10000

  config = Qwen2Config(
    hidden_size=hidden_size,
    num_hidden_layers=2,
    intermediate_size=hidden_size * 4,
    num_attention_heads=num_heads,
    num_key_value_heads=num_kv_heads,
    rms_norm_eps=1e-6,
    vocab_size=1000,
    rope_theta=theta,
    max_position_embeddings=max_seq_len,
  )
  config._attn_implementation = "eager"
  rotary_emb = Qwen2RotaryEmbedding(config)
  torch_attention = Qwen2Attention(config, layer_idx=0).to(device=dev, dtype=dtype)

  torch.manual_seed(42)
  x = torch.rand(batch_size, seq_len, hidden_size, dtype=dtype, device=dev) * 2 - 1
  position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)
  # Prepare attention mask for torch attention
  if mask == "causal":
    attention_mask = torch.triu(
      torch.ones(seq_len, seq_len, dtype=torch.bool, device=dev), diagonal=1
    )
    attention_mask = (
      attention_mask.masked_fill(attention_mask, float("-inf")).unsqueeze(0).unsqueeze(0)
    )
  else:
    attention_mask = None

  position_embeddings = rotary_emb(x, position_ids)
  torch_output = torch_attention(x, position_embeddings, attention_mask=attention_mask)[0]
  print(torch_output)


if __name__ == "__main__":
  test_qwen2_gqa_attention(
    mask="causal",
    dtype=torch.float32,
    device="cuda",
  )
