import torch
from transformers import AutoModelForCausalLM
from miniinfer.config.model.qwen2 import Qwen2Config
from miniinfer.loader.weight import load_hf_weight
from miniinfer.models.fused_qwen2 import Qwen2ForCausalLM
from tests.utils import create_simple_forward_batch

model_name = "Qwen/Qwen2-0.5B-Instruct"
device = torch.device("cuda")

# 加载模型
print("Loading models...")
hf_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map=device)
config = Qwen2Config.from_pretrained(model_name)
my_model = Qwen2ForCausalLM(config=config, device=device, precision=torch.float16)
state_dict = load_hf_weight(model_name, device=device)
my_model.load_weights(state_dict)

# 测试数据
torch.manual_seed(42)
batch_size, seq_len = 1, 3
input_ids = torch.tensor([[643, 409, 798]], device=device)
positions = torch.arange(seq_len, device=device)

forward_batch = create_simple_forward_batch(
  batch_size=batch_size,
  seq_len=seq_len,
  input_ids=input_ids,
  positions=positions,
  device=device,
  use_causal_mask=True,
)

print(f"Input IDs: {input_ids}")
print()

# ==============================================================================
# Embedding
# ==============================================================================
print("=" * 80)
print("EMBEDDING")
print("=" * 80)
with torch.no_grad():
  hf_embed = hf_model.model.embed_tokens(input_ids)
  my_embed = my_model.qwen2.embedding(input_ids.reshape(-1)).reshape(batch_size, seq_len, -1)

print(f"HF: {hf_embed[0, 0, :5]}")
print(f"My: {my_embed[0, 0, :5]}")
print(f"Diff: {torch.abs(hf_embed - my_embed).max().item()}")
print()

# ==============================================================================
# Layer 0
# ==============================================================================
print("=" * 80)
print("LAYER 0")
print("=" * 80)

# HF
with torch.no_grad():
  hf_hidden = hf_embed
  hf_residual = hf_hidden
  hf_hidden_norm = hf_model.model.layers[0].input_layernorm(hf_hidden)
  print(f"HF input_layernorm output: {hf_hidden_norm[0, 0, :5]}")

# My
my_hidden = my_embed.reshape(-1, config.hidden_size)
my_residual = None
if my_residual is None:
  my_residual = my_hidden.clone()
  my_hidden_norm = my_model.qwen2.layers[0].input_layernorm(my_hidden)
else:
  my_hidden_norm, my_residual = my_model.qwen2.layers[0].input_layernorm(my_hidden, my_residual)
my_hidden_norm_view = my_hidden_norm.reshape(batch_size, seq_len, -1)
print(f"My input_layernorm output: {my_hidden_norm_view[0, 0, :5]}")
print(f"Diff: {torch.abs(hf_hidden_norm - my_hidden_norm_view).max().item()}")
print()

# After attention
with torch.no_grad():
  hf_position_embeddings = hf_model.model.rotary_emb(
    hf_hidden_norm, position_ids=positions.unsqueeze(0)
  )
  hf_attn_out, _ = hf_model.model.layers[0].self_attn(
    hf_hidden_norm,
    attention_mask=None,
    position_embeddings=hf_position_embeddings,
  )
  print(f"HF attention output: {hf_attn_out[0, 0, :5]}")
  hf_hidden = hf_residual + hf_attn_out
  print(f"HF after add residual: {hf_hidden[0, 0, :5]}")

my_attn_out = my_model.qwen2.layers[0].self_attn(
  positions=positions,
  hidden_states=my_hidden_norm,
  forward_batch=forward_batch,
)
my_attn_out_view = my_attn_out.reshape(batch_size, seq_len, -1)
print(f"My attention output: {my_attn_out_view[0, 0, :5]}")
print(f"Attention diff: {torch.abs(hf_attn_out - my_attn_out_view).max().item()}")

# Post attention layernorm
my_hidden_norm2, my_residual = my_model.qwen2.layers[0].post_attention_layernorm(
  my_attn_out, my_residual
)
my_residual_view = my_residual.reshape(batch_size, seq_len, -1)
print(f"My after post_attn norm residual: {my_residual_view[0, 0, :5]}")
print(f"Residual diff: {torch.abs(hf_hidden - my_residual_view).max().item()}")
print()

# Post attention norm output
with torch.no_grad():
  hf_residual2 = hf_hidden
  hf_hidden_norm2 = hf_model.model.layers[0].post_attention_layernorm(hf_hidden)
  print(f"HF post_attn_layernorm output: {hf_hidden_norm2[0, 0, :5]}")

my_hidden_norm2_view = my_hidden_norm2.reshape(batch_size, seq_len, -1)
print(f"My post_attn_layernorm output: {my_hidden_norm2_view[0, 0, :5]}")
print(f"Post norm diff: {torch.abs(hf_hidden_norm2 - my_hidden_norm2_view).max().item()}")
print()

# MLP
with torch.no_grad():
  hf_mlp_out = hf_model.model.layers[0].mlp(hf_hidden_norm2)
  print(f"HF MLP output: {hf_mlp_out[0, 0, :5]}")
  hf_hidden = hf_residual2 + hf_mlp_out
  print(f"HF after MLP + residual: {hf_hidden[0, 0, :5]}")

my_mlp_out = my_model.qwen2.layers[0].mlp(my_hidden_norm2)
my_mlp_out_view = my_mlp_out.reshape(batch_size, seq_len, -1)
print(f"My MLP output: {my_mlp_out_view[0, 0, :5]}")
print(f"MLP diff: {torch.abs(hf_mlp_out - my_mlp_out_view).max().item()}")

# Our layer output
print(f"My layer output (MLP out): {my_mlp_out_view[0, 0, :5]}")
print(f"My layer residual: {my_residual_view[0, 0, :5]}")
print(f"My final (mlp + residual): {(my_mlp_out_view + my_residual_view)[0, 0, :5]}")
print(
  f"Final layer diff: {torch.abs(hf_hidden - (my_mlp_out_view + my_residual_view)).max().item()}"
)
