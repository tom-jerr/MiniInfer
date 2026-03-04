"""
Debug script to compare MiniInfer vs HuggingFace layer by layer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "miniinfer"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from config.engine.config import EngineConfig
from engine.model_runner import ModelRunner
from kvcache.kv_cache_manager import KVCacheManager
from layers.attention_backend.flashattention_backend import FlashAttention2Backend
from scheduler.scheduler_batch import Req, ScheduledBatch, ForwardBatch
from utils.sampling_params import SamplingParams


def compare_weights(model_path: str, device: str = "cuda"):
  """Compare weight loading between MiniInfer and HuggingFace."""
  print("=" * 60)
  print("Comparing weights between MiniInfer and HuggingFace")
  print("=" * 60)

  # Load HuggingFace model
  hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
  ).to(device)
  hf_model.eval()

  # Load MiniInfer model
  from loader.weight import load_hf_weight
  from models.fused_qwen2 import Qwen2ForCausalLM
  from transformers import Qwen2Config

  config = Qwen2Config.from_pretrained(model_path)
  state_dict = load_hf_weight(model_path, device)

  mi_model = Qwen2ForCausalLM(config, device=device, precision=torch.float16)
  mi_model.load_weights(state_dict)
  mi_model.eval()

  # Compare embeddings
  print("\n--- Embedding weights ---")
  hf_emb = hf_model.model.embed_tokens.weight.data
  mi_emb = mi_model.qwen2.embedding.weight
  diff = (hf_emb - mi_emb).abs()
  print(f"  Shape HF: {hf_emb.shape}, MI: {mi_emb.shape}")
  print(f"  Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")

  # Compare lm_head
  print("\n--- LM Head weights ---")
  hf_lmhead = hf_model.lm_head.weight.data
  mi_lmhead = mi_model.lm_head.weight
  diff = (hf_lmhead - mi_lmhead).abs()
  print(f"  Shape HF: {hf_lmhead.shape}, MI: {mi_lmhead.shape}")
  print(f"  Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")

  # Compare first layer weights
  print("\n--- Layer 0 self_attn weights ---")
  hf_layer0 = hf_model.model.layers[0]
  mi_layer0 = mi_model.qwen2.layers[0]

  # Q, K, V weights
  hf_q = hf_layer0.self_attn.q_proj.weight.data
  hf_k = hf_layer0.self_attn.k_proj.weight.data
  hf_v = hf_layer0.self_attn.v_proj.weight.data
  hf_qkv = torch.cat([hf_q, hf_k, hf_v], dim=0)

  mi_qkv = mi_layer0.self_attn.qkv_proj.weight.data
  print(f"  HF QKV shape: {hf_qkv.shape}, MI QKV shape: {mi_qkv.shape}")
  diff = (hf_qkv - mi_qkv).abs()
  print(f"  QKV weight max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  # Q, K, V biases
  hf_q_bias = hf_layer0.self_attn.q_proj.bias.data
  hf_k_bias = hf_layer0.self_attn.k_proj.bias.data
  hf_v_bias = hf_layer0.self_attn.v_proj.bias.data
  hf_qkv_bias = torch.cat([hf_q_bias, hf_k_bias, hf_v_bias], dim=0)

  mi_qkv_bias = mi_layer0.self_attn.qkv_proj.bias.data
  print(f"  HF QKV bias shape: {hf_qkv_bias.shape}, MI QKV bias shape: {mi_qkv_bias.shape}")
  diff = (hf_qkv_bias - mi_qkv_bias).abs()
  print(f"  QKV bias max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  # O_proj
  hf_o = hf_layer0.self_attn.o_proj.weight.data
  mi_o = mi_layer0.self_attn.o_proj.weight.data
  diff = (hf_o - mi_o).abs()
  print(f"  O_proj weight max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  # MLP weights
  print("\n--- Layer 0 MLP weights ---")
  hf_gate = hf_layer0.mlp.gate_proj.weight.data
  hf_up = hf_layer0.mlp.up_proj.weight.data
  hf_gate_up = torch.cat([hf_gate, hf_up], dim=0)

  mi_gate_up = mi_layer0.mlp.gate_up_proj.weight.data
  diff = (hf_gate_up - mi_gate_up).abs()
  print(f"  Gate+Up weight max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  hf_down = hf_layer0.mlp.down_proj.weight.data
  mi_down = mi_layer0.mlp.down_proj.weight.data
  diff = (hf_down - mi_down).abs()
  print(f"  Down weight max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  # LayerNorm weights
  print("\n--- Layer 0 LayerNorm weights ---")
  hf_ln1 = hf_layer0.input_layernorm.weight.data
  mi_ln1 = mi_layer0.input_layernorm.weight
  diff = (hf_ln1 - mi_ln1).abs()
  print(f"  Input LN max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  hf_ln2 = hf_layer0.post_attention_layernorm.weight.data
  mi_ln2 = mi_layer0.post_attention_layernorm.weight
  diff = (hf_ln2 - mi_ln2).abs()
  print(f"  Post-attn LN max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  # Final norm
  print("\n--- Final norm weights ---")
  hf_final = hf_model.model.norm.weight.data
  mi_final = mi_model.qwen2.norm.weight
  diff = (hf_final - mi_final).abs()
  print(f"  Final norm max diff: {diff.max().item():.6e}, mean diff: {diff.mean().item():.6e}")

  return hf_model, mi_model


def compare_forward(hf_model, mi_model, input_ids, device: str = "cuda"):
  """Compare forward pass outputs."""
  print("\n" + "=" * 60)
  print("Comparing forward pass")
  print("=" * 60)

  input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

  # HuggingFace forward
  with torch.no_grad():
    hf_out = hf_model(input_ids=input_tensor)
    hf_logits = hf_out.logits[0, -1]  # Last token logits

  # MiniInfer forward requires ForwardBatch setup
  # We'll do a simpler comparison: just run the embedding
  print("\n--- Embedding output ---")
  hf_emb_out = hf_model.model.embed_tokens(input_tensor)
  mi_emb_out = mi_model.qwen2.embedding(input_tensor.squeeze(0))

  diff = (hf_emb_out.squeeze(0) - mi_emb_out).abs()
  print(f"  Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")

  # Compare RMSNorm output
  print("\n--- Layer 0 input_layernorm output ---")
  hf_ln_out = hf_model.model.layers[0].input_layernorm(hf_emb_out)
  mi_ln_out = mi_model.qwen2.layers[0].input_layernorm(mi_emb_out)

  diff = (hf_ln_out.squeeze(0) - mi_ln_out).abs()
  print(f"  Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")

  # Compare QKV projection
  print("\n--- Layer 0 QKV projection output ---")
  hf_layer0 = hf_model.model.layers[0]
  mi_layer0 = mi_model.qwen2.layers[0]

  hf_q = hf_layer0.self_attn.q_proj(hf_ln_out)
  hf_k = hf_layer0.self_attn.k_proj(hf_ln_out)
  hf_v = hf_layer0.self_attn.v_proj(hf_ln_out)

  mi_qkv = mi_layer0.self_attn.qkv_proj(mi_ln_out)
  mi_q, mi_k, mi_v = mi_qkv.split(
    [
      mi_layer0.self_attn.q_size,
      mi_layer0.self_attn.kv_size,
      mi_layer0.self_attn.kv_size,
    ],
    dim=-1,
  )

  diff_q = (hf_q.squeeze(0) - mi_q).abs()
  diff_k = (hf_k.squeeze(0) - mi_k).abs()
  diff_v = (hf_v.squeeze(0) - mi_v).abs()
  print(f"  Q max diff: {diff_q.max().item():.6e}, mean diff: {diff_q.mean().item():.6e}")
  print(f"  K max diff: {diff_k.max().item():.6e}, mean diff: {diff_k.mean().item():.6e}")
  print(f"  V max diff: {diff_v.max().item():.6e}, mean diff: {diff_v.mean().item():.6e}")

  # Compare RoPE
  print("\n--- Layer 0 RoPE output ---")
  seq_len = input_tensor.shape[1]
  positions = torch.arange(seq_len, device=device)

  # Get config from model
  config = hf_model.config

  # MiniInfer RoPE
  mi_q_reshaped = mi_q.view(seq_len, mi_layer0.self_attn.num_heads, mi_layer0.self_attn.head_dim)
  mi_k_reshaped = mi_k.view(seq_len, mi_layer0.self_attn.num_kv_heads, mi_layer0.self_attn.head_dim)
  mi_q_rope, mi_k_rope = mi_layer0.self_attn.rope(positions, mi_q_reshaped, mi_k_reshaped)

  # HuggingFace RoPE
  hf_q_reshaped = hf_q.view(
    1,
    seq_len,
    config.num_attention_heads,
    config.hidden_size // config.num_attention_heads,
  ).transpose(1, 2)
  hf_k_reshaped = hf_k.view(
    1,
    seq_len,
    config.num_key_value_heads,
    config.hidden_size // config.num_attention_heads,
  ).transpose(1, 2)

  cos, sin = hf_layer0.self_attn.rotary_emb(hf_v, positions.unsqueeze(0))
  from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

  hf_q_rope, hf_k_rope = apply_rotary_pos_emb(hf_q_reshaped, hf_k_reshaped, cos, sin)

  # Reshape for comparison
  hf_q_rope = hf_q_rope.transpose(1, 2).squeeze(0).reshape(seq_len, -1)
  hf_k_rope = hf_k_rope.transpose(1, 2).squeeze(0).reshape(seq_len, -1)
  mi_q_rope = mi_q_rope.reshape(seq_len, -1)
  mi_k_rope = mi_k_rope.reshape(seq_len, -1)

  diff_q = (hf_q_rope - mi_q_rope).abs()
  diff_k = (hf_k_rope - mi_k_rope).abs()
  print(
    f"  Q after RoPE max diff: {diff_q.max().item():.6e}, mean diff: {diff_q.mean().item():.6e}"
  )
  print(
    f"  K after RoPE max diff: {diff_k.max().item():.6e}, mean diff: {diff_k.mean().item():.6e}"
  )


def main():
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", required=True, help="Model path")
  args = parser.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device: {device}")

  torch.set_grad_enabled(False)

  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

  # Simple test input
  test_prompt = "Hello"
  input_ids = tokenizer.encode(test_prompt)
  print(f"Test input: {test_prompt!r} -> {input_ids}")

  hf_model, mi_model = compare_weights(args.model, device)
  compare_forward(hf_model, mi_model, input_ids, device)


if __name__ == "__main__":
  main()
