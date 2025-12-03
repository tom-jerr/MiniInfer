以 Qwen2 模型为例，在 SGLang 的推理过程中，RoPE、RMSNorm 以及算子融合的情况如下：

1. RoPE (Rotary Positional Embedding) 的执行位置
RoPE 是在 QKV 投影之后，Attention 计算之前 执行的。

代码位置: qwen2.py 中的 Qwen2Attention.forward 方法。
执行流程:
qkv_proj: 首先对输入 hidden_states 进行线性投影得到 q, k, v。
RoPE: 调用 self.rotary_emb(positions, q, k)。
底层调用的是 rotary_embedding.py 中的 RotaryEmbedding.forward。
在 CUDA 设备上，最终调用 sgl_kernel 中的 apply_rope_with_cos_sin_cache_inplace 算子。这是一个 In-place 操作，直接修改显存中的 q 和 k。
attn: 经过 RoPE 处理后的 q, k 以及原始的 v 被送入 RadixAttention 进行注意力计算。
2. RMSNorm 的执行方式
RMSNorm 通常与残差连接（Residual Add）融合执行。

代码位置: layernorm.py 中的 RMSNorm.forward。
执行流程:
在 Transformer Layer 中（如 Qwen2DecoderLayer），通常会有 hidden_states, residual = self.input_layernorm(hidden_states, residual) 这样的调用。
如果传入了 residual 参数，SGLang 会调用融合算子 fused_add_rmsnorm。
该算子在一个 CUDA Kernel 中同时完成：
input + residual (残差相加)
RMSNorm(...) (归一化)
更新 residual (用于下一次残差连接)
3. 哪些算子是融合在一块执行的？
在 Qwen2 的标准推理路径中，主要有以下几个融合算子：

Fused Add + RMSNorm:

算子: fused_add_rmsnorm
作用: 将残差相加和 RMSNorm 归一化融合，减少显存读写。
位置: 每个 Transformer Layer 的开始（Pre-Norm）和 Attention/MLP 之后（Post-Norm）。
Fused SiLU + Multiply (SwiGLU):

算子: silu_and_mul
作用: Qwen2 的 MLP 层使用 SwiGLU 激活函数。该算子将 SiLU(gate) * up 的计算融合，避免中间结果写回显存。
位置: Qwen2MLP 中的 self.act_fn。

**支持 rope + kvcache copy 的 Model**
根据代码搜索结果，以下模型的 Python 定义层面明确支持 RoPE 和 KV Cache Copy 的融合（即使用了 fused_set_kv_buffer_arg 参数）：

Llada2 (llada2.py)
Bailing MoE (bailing_moe.py)
Qwen3 MoE (qwen3_moe.py)
GPT-OSS (gpt_oss.py)