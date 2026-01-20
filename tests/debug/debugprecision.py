import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2Config as HFQwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention as HFQwen2Attention,
    Qwen2MLP as HFQwen2MLP,
    Qwen2RMSNorm as HFQwen2RMSNorm,
    Qwen2DecoderLayer as HFQwen2DecoderLayer,
    Qwen2RotaryEmbedding as HFRoPE,
)

from miniinfer.config.model.qwen2 import Qwen2Config
from miniinfer.loader.weight import load_hf_weight
from miniinfer.models.fused_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2TransformerBlock,
    Qwen2ForCausalLM,
)
from miniinfer.layers import RMSNorm, RotaryEmbedding
from tests.utils import create_simple_forward_batch, assert_allclose


def test_rope(model_name: str, device, layer_idx=0):
    """测试 RoPE 旋转位置编码"""
    print("\n" + "=" * 80)
    print("测试 RoPE (Rotary Position Embedding)")
    print("=" * 80)

    config = Qwen2Config.from_pretrained(model_name)
    head_dim = config.hidden_size // config.num_attention_heads

    hf_config = HFQwen2Config.from_pretrained(model_name)
    hf_rope = HFRoPE(config=hf_config, device=device)

    # Our RoPE
    my_rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
        is_neox_style=True,
        dtype=torch.float16,
    )

    # Test data
    batch_size, seq_len = 1, 10
    positions = torch.arange(seq_len, device=device)
    q = torch.randn(
        batch_size,
        seq_len,
        config.num_attention_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    k = torch.randn(
        batch_size,
        seq_len,
        config.num_key_value_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )

    # HF implementation
    cos, sin = hf_rope(k, position_ids=positions.unsqueeze(0))

    # cos/sin shape: [batch, seq_len, head_dim] need to be [batch, seq_len, 1, head_dim]
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    print(f"After unsqueeze - cos shape: {cos.shape}, sin shape: {sin.shape}")
    print(f"q shape: {q.shape}, k shape: {k.shape}")

    def apply_rotary_pos_emb(q, k, cos, sin):
        # q: [batch, seq_len, num_heads, head_dim]
        # k: [batch, seq_len, num_kv_heads, head_dim]
        # cos/sin: [batch, seq_len, 1, head_dim]
        q_embed = (q * cos) + (
            torch.cat([-q[..., q.shape[-1] // 2 :], q[..., : q.shape[-1] // 2]], dim=-1)
            * sin
        )
        k_embed = (k * cos) + (
            torch.cat([-k[..., k.shape[-1] // 2 :], k[..., : k.shape[-1] // 2]], dim=-1)
            * sin
        )
        return q_embed, k_embed

    hf_q, hf_k = apply_rotary_pos_emb(q, k, cos, sin)

    # Our implementation
    q_flat = q.reshape(-1, config.num_attention_heads * head_dim)
    k_flat = k.reshape(-1, config.num_key_value_heads * head_dim)
    my_q, my_k = my_rope(positions, q_flat, k_flat)
    my_q = my_q.reshape(batch_size, seq_len, config.num_attention_heads, head_dim)
    my_k = my_k.reshape(batch_size, seq_len, config.num_key_value_heads, head_dim)

    print(f"HF RoPE Q shape: {hf_q.shape}, Our RoPE Q shape: {my_q.shape}")
    print(f"HF RoPE Q sample: {hf_q[0, 0, 0, :5]}")
    print(f"Our RoPE Q sample: {my_q[0, 0, 0, :5]}")

    try:
        assert_allclose(my_q, hf_q, precision=torch.float16, rtol=1e-2)
        assert_allclose(my_k, hf_k, precision=torch.float16, rtol=1e-2)
        print("✓ RoPE 精度测试通过")
        return True
    except Exception as e:
        print(f"✗ RoPE 精度测试失败: {e}")
        return False


def test_rmsnorm(model_name: str, device, layer_idx=0):
    """测试 RMSNorm"""
    print("\n" + "=" * 80)
    print("测试 RMSNorm")
    print("=" * 80)

    config = Qwen2Config.from_pretrained(model_name)
    state_dict = load_hf_weight(model_name, device=device)

    # Load weights
    has_model_prefix = "model.embed_tokens.weight" in state_dict
    prefix = f"{'model.' if has_model_prefix else ''}layers.{layer_idx}.input_layernorm"

    # HF RMSNorm
    hf_norm = HFQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
        device, torch.float16
    )
    hf_norm.weight.data = state_dict[f"{prefix}.weight"].to(device, torch.float16)

    # Our RMSNorm
    my_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    my_norm.load_weights(state_dict, prefix, device, torch.float16)

    # Test data
    x = torch.randn(1, 10, config.hidden_size, dtype=torch.float16, device=device)

    with torch.no_grad():
        hf_output = hf_norm(x)
        my_output = my_norm(x.reshape(-1, config.hidden_size)).reshape(1, 10, -1)

    print(f"HF RMSNorm output shape: {hf_output.shape}")
    print(f"Our RMSNorm output shape: {my_output.shape}")
    print(f"HF RMSNorm sample: {hf_output[0, 0, :5]}")
    print(f"Our RMSNorm sample: {my_output[0, 0, :5]}")

    try:
        assert_allclose(my_output, hf_output, precision=torch.float16, rtol=1e-2)
        print("✓ RMSNorm 精度测试通过")
        return True
    except Exception as e:
        print(f"✗ RMSNorm 精度测试失败: {e}")
        return False


def test_mlp(model_name: str, device, layer_idx=0):
    """测试 MLP"""
    print("\n" + "=" * 80)
    print("测试 MLP")
    print("=" * 80)

    config = Qwen2Config.from_pretrained(model_name)
    state_dict = load_hf_weight(model_name, device=device)

    has_model_prefix = "model.embed_tokens.weight" in state_dict
    prefix = f"{'model.' if has_model_prefix else ''}layers.{layer_idx}.mlp"

    # HF MLP
    hf_config = HFQwen2Config.from_pretrained(model_name)
    hf_mlp = HFQwen2MLP(hf_config).to(device, torch.float16)
    hf_mlp.gate_proj.weight.data = state_dict[f"{prefix}.gate_up_proj.weight"][
        : config.intermediate_size
    ].to(device, torch.float16)
    hf_mlp.up_proj.weight.data = state_dict[f"{prefix}.gate_up_proj.weight"][
        config.intermediate_size :
    ].to(device, torch.float16)
    hf_mlp.down_proj.weight.data = state_dict[f"{prefix}.down_proj.weight"].to(
        device, torch.float16
    )

    # Our MLP
    my_mlp = Qwen2MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    ).to(device, torch.float16)
    my_mlp.load_weights(state_dict, prefix, device, torch.float16)

    # Test data
    x = torch.randn(10, config.hidden_size, dtype=torch.float16, device=device)

    with torch.no_grad():
        hf_output = hf_mlp(x.unsqueeze(0)).squeeze(0)
        my_output = my_mlp(x)

    print(f"HF MLP output shape: {hf_output.shape}")
    print(f"Our MLP output shape: {my_output.shape}")
    print(f"HF MLP sample: {hf_output[0, :5]}")
    print(f"Our MLP sample: {my_output[0, :5]}")

    try:
        assert_allclose(my_output, hf_output, precision=torch.float16, rtol=1e-1)
        print("✓ MLP 精度测试通过")
        return True
    except Exception as e:
        print(f"✗ MLP 精度测试失败: {e}")
        return False


def test_attention(model_name: str, device, layer_idx=0):
    """测试 Attention"""
    print("\n" + "=" * 80)
    print("测试 Attention")
    print("=" * 80)

    config = Qwen2Config.from_pretrained(model_name)
    state_dict = load_hf_weight(model_name, device=device)

    has_model_prefix = "model.embed_tokens.weight" in state_dict
    prefix = f"{'model.' if has_model_prefix else ''}layers.{layer_idx}.self_attn"

    # HF Attention
    hf_config = HFQwen2Config.from_pretrained(model_name)
    hf_config._attn_implementation = "eager"
    hf_attn = HFQwen2Attention(hf_config, layer_idx=layer_idx).to(device, torch.float16)
    hf_rope = HFRoPE(config=hf_config, device=device)

    # Load original weights (not merged)
    original_state = load_hf_weight.__code__
    # Re-load without merging for HF model
    from miniinfer.loader.weight import load_hf_weight as load_func
    import glob
    import safetensors
    from huggingface_hub import snapshot_download

    hf_folder = snapshot_download(model_name, allow_patterns=["*.safetensors"])
    files = glob.glob(f"{hf_folder}/*.safetensors")
    raw_state_dict = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                raw_state_dict[name] = f.get_tensor(name)

    p = "model." if has_model_prefix else ""
    hf_attn.q_proj.weight.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.q_proj.weight"
    ].to(device, torch.float16)
    hf_attn.q_proj.bias.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.q_proj.bias"
    ].to(device, torch.float16)
    hf_attn.k_proj.weight.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.k_proj.weight"
    ].to(device, torch.float16)
    hf_attn.k_proj.bias.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.k_proj.bias"
    ].to(device, torch.float16)
    hf_attn.v_proj.weight.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.v_proj.weight"
    ].to(device, torch.float16)
    hf_attn.v_proj.bias.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.v_proj.bias"
    ].to(device, torch.float16)
    hf_attn.o_proj.weight.data = raw_state_dict[
        f"{p}layers.{layer_idx}.self_attn.o_proj.weight"
    ].to(device, torch.float16)

    # Our Attention
    my_attn = Qwen2Attention(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        layer_id=layer_idx,
        rope_theta=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
    ).to(device, torch.float16)
    my_attn.load_weights(state_dict, prefix, device, torch.float16)

    # Test data
    batch_size, seq_len = 1, 10
    x = torch.randn(
        batch_size, seq_len, config.hidden_size, dtype=torch.float16, device=device
    )
    positions = torch.arange(seq_len, device=device)

    forward_batch = create_simple_forward_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        input_ids=x,
        positions=positions,
        device=device,
    )

    with torch.no_grad():
        hf_rope_output = hf_rope(x, position_ids=positions.unsqueeze(0))
        hf_output = hf_attn(
            x,
            position_embeddings=hf_rope_output,
            attention_mask=None,
        )[0]
        my_output = my_attn(positions, x.reshape(-1, config.hidden_size), forward_batch)
        my_output = my_output.reshape(batch_size, seq_len, -1)

    print(f"HF Attention output shape: {hf_output.shape}")
    print(f"Our Attention output shape: {my_output.shape}")
    print(f"HF Attention sample: {hf_output[0, 0, :5]}")
    print(f"Our Attention sample: {my_output[0, 0, :5]}")

    try:
        assert_allclose(
            my_output, hf_output, precision=torch.float16, rtol=2e-1, atol=2e-1
        )
        print("✓ Attention 精度测试通过")
        return True
    except Exception as e:
        print(f"✗ Attention 精度测试失败: {e}")
        return False


def test_full_model(model_name: str, device):
    """测试完整模型"""
    print("\n" + "=" * 80)
    print("测试完整模型")
    print("=" * 80)

    # HF model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Our model
    config = Qwen2Config.from_pretrained(model_name)
    my_model = Qwen2ForCausalLM(config=config, device=device, precision=torch.float16)
    state_dict = load_hf_weight(model_name, device=device)
    my_model.load_weights(state_dict)

    # Test data
    batch_size, seq_len = 1, 10
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len), device=device
    )
    positions = torch.arange(seq_len, device=device)

    forward_batch = create_simple_forward_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        input_ids=input_ids,
        positions=positions,
        device=device,
        use_causal_mask=True,
    )

    with torch.no_grad():
        hf_output = hf_model(input_ids).logits
        my_output = my_model(
            input_ids, positions=positions, forward_batch=forward_batch
        ).logits
        my_output = my_output.reshape(batch_size, seq_len, -1)

    print(f"HF Model output shape: {hf_output.shape}")
    print(f"Our Model output shape: {my_output.shape}")
    print(f"HF Model sample: {hf_output[0, 0, :5]}")
    print(f"Our Model sample: {my_output[0, 0, :5]}")

    # Apply log softmax
    hf_output = hf_output - torch.logsumexp(hf_output, dim=-1, keepdim=True)
    my_output = my_output - torch.logsumexp(my_output, dim=-1, keepdim=True)

    try:
        assert_allclose(
            my_output, hf_output, precision=torch.float16, rtol=1e-1, atol=1e-1
        )
        print("✓ 完整模型精度测试通过")
        return True
    except Exception as e:
        print(f"✗ 完整模型精度测试失败: {e}")
        return False


if __name__ == "__main__":
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"测试模型: {model_name}")

    results = {}
    # results["RoPE"] = test_rope(model_name, device, layer_idx=0)
    # results["RMSNorm"] = test_rmsnorm(model_name, device, layer_idx=0)
    # results["MLP"] = test_mlp(model_name, device, layer_idx=0)
    # results["Attention"] = test_attention(model_name, device, layer_idx=0)
    results["FullModel"] = test_full_model(model_name, device)

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
