import torch
import triton
import triton.language as tl

@triton.jit
def store_kv_cache_kernel(
    k_ptr, v_ptr,           # 输入的 K, V Tensor (当前 Step) [L, D]
    k_cache_ptr, v_cache_ptr, # KV Cache 池 [Max_Blocks, D] 或类似扁平化结构
    indices_ptr,            # 写入的目标位置索引 [L]
    stride_k_l, stride_k_d,  # K 输入的步长
    stride_cache_x, stride_cache_d, # Cache 的步长
    L, D,                   # 序列长度和 Head Dim
    BLOCK_D: tl.constexpr,   # 维度分块大小 (通常设为 head_dim 的下一个 2 的幂)
):
    # 每个 Program 处理序列 L 中的一个 token (对应 CUDA 中的一个 Warp)
    token_idx = tl.program_id(0)
    
    if token_idx < L:
        # 1. 获取该 token 对应的目标物理位置
        target_pos = tl.load(indices_ptr + token_idx)
        
        # 2. 计算输入和输出的偏移量
        # 输入偏移: token_idx * stride_k_l
        # 输出偏移: target_pos * stride_cache_x
        offsets_d = tl.arange(0, BLOCK_D)
        mask_d = offsets_d < D
        
        k_in_offsets = token_idx * stride_k_l + offsets_d * stride_k_d
        v_in_offsets = token_idx * stride_k_l + offsets_d * stride_k_d # 假设 v 和 k 步长一致
        
        k_cache_offsets = target_pos * stride_cache_x + offsets_d * stride_cache_d
        v_cache_offsets = target_pos * stride_cache_x + offsets_d * stride_cache_d
        
        # 3. 加载并存储
        k_val = tl.load(k_ptr + k_in_offsets, mask=mask_d)
        v_val = tl.load(v_ptr + v_in_offsets, mask=mask_d)
        
        tl.store(k_cache_ptr + k_cache_offsets, k_val, mask=mask_d)
        tl.store(v_cache_ptr + v_cache_offsets, v_val, mask=mask_d)

def store_kv_cache(k_cache, v_cache, indices, k, v):
    # 简单的输入校验
    L, D = k.shape
    
    # 确定 Triton 的 BLOCK_SIZE
    # 假设 D (head_dim) 不会太大（如 64, 128, 256），可以直接用一个 block 处理完一整行
    BLOCK_D = triton.next_power_of_2(D)
    
    grid = (L,)
    
    store_kv_cache_kernel[grid](
        k, v,
        k_cache, v_cache,
        indices,
        k.stride(0), k.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        L, D,
        BLOCK_D=BLOCK_D,
    )