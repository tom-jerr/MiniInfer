"""简化版 fused-MoE Triton kernel。

移植自 vLLM 的 ``fused_moe.py`` / nano-vllm，去掉所有不需要的部分：

- 无量化路径（fp8/int8/int4）；仅 BF16/FP16/FP32
- 无 bias
- 无 SPLIT_K
- 无 naive_block_assignment
- 无 GPTQ/AWQ kernel

剩余逻辑是标准的 sorted-token grouped GEMM：

    moe_align_block_size(topk_ids, BLOCK_M, num_experts)
        -> sorted_token_ids, expert_ids, num_tokens_post_padded

    fused_moe_kernel
        -> 对每个 (pid_m, pid_n) tile，查 expert_ids[pid_m]，把
           A[sorted_token_ids[..]/top_k] @ B[expert] 累加进 C[..]。

启动两段 GEMM：先 W13（gate||up），再 W2，第二段 ``mul_routed_weight=True``
把 topk 权重折进最终 reduce。

Python 入口：:func:`fused_experts_triton`。
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helper: pure-Python moe_align_block_size（无 C++ 依赖，作参考实现）。
# ---------------------------------------------------------------------------
def moe_align_block_size_torch(
    topk_ids: torch.Tensor,         # (T, K) int32/int64
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把 (token, k) 对按专家分组，每个专家桶补齐到 ``block_size`` 的倍数。

    Returns
    -------
    sorted_token_ids : (M_pad,) int32 —— A_perm 每一行对应的原始
        flat (token*K + k) 索引（补齐位用哨兵 = T*K）。
    expert_ids : (M_pad // block_size,) int32 —— 每个 BLOCK_M tile 的专家。
    num_tokens_post_padded : (1,) int32 标量。
    """
    T, K = topk_ids.shape
    flat_expert = topk_ids.reshape(-1).to(torch.int64)              # (T*K,)
    flat_pair = torch.arange(T * K, device=topk_ids.device, dtype=torch.int32)

    # 按专家稳定排序。
    order = torch.argsort(flat_expert, stable=True)
    sorted_expert = flat_expert[order]
    sorted_pair = flat_pair[order]

    # 每个专家的 token 计数。
    counts = torch.bincount(sorted_expert, minlength=num_experts)   # (E,)
    padded = ((counts + block_size - 1) // block_size) * block_size  # (E,)
    num_blocks_per_expert = padded // block_size                    # (E,)
    M_pad = int(padded.sum().item())
    num_blocks = M_pad // block_size

    SENTINEL = T * K
    sorted_token_ids = torch.full(
        (M_pad,), SENTINEL, dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=topk_ids.device)

    # 逐专家填充（CPU 侧循环没问题：E 很小，≤ ~256）。
    src_offset = 0
    dst_offset = 0
    block_offset = 0
    counts_list = counts.tolist()
    padded_list = padded.tolist()
    nblocks_list = num_blocks_per_expert.tolist()
    for e in range(num_experts):
        n = counts_list[e]
        p = padded_list[e]
        if n > 0:
            sorted_token_ids[dst_offset : dst_offset + n] = sorted_pair[
                src_offset : src_offset + n
            ]
        src_offset += n
        dst_offset += p
        nb = nblocks_list[e]
        if nb:
            expert_ids[block_offset : block_offset + nb] = e
            block_offset += nb

    num_tokens_post_padded = torch.tensor(
        [M_pad], dtype=torch.int32, device=topk_ids.device
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ---------------------------------------------------------------------------
# Triton-fused moe_align_block_size（4-kernel 流水，无 host sync）。
#
# 算法（镜像 vLLM CUDA `_moe_align_block_size`）：
#
#   A) _count_kernel:     对每个 (token, k) atomic-add counts[expert]。
#   B) _meta_kernel:      grid=(1,)，单 block 在 BLOCK_E = next_pow2(E) 条
#                         lane 上算 padded[e]=ceil(counts[e]/B)*B、exclusive
#                         cumsum -> base[e]，同时写 counter[e]=base[e]（供
#                         后续 atomic-scatter）和 num_tokens_post_padded。
#   C) _fill_kernel:      grid=(E, num_chunks)；每个专家 e 填
#                         expert_ids[base[e]/B : base[e]/B + nb_e] = e，以及
#                         sorted_token_ids 的哨兵补齐区 [base[e]+counts[e],
#                         base[e]+padded[e])。
#   D) _scatter_kernel:   atomic-add counter[expert] -> 桶内 rank；
#                         sorted_token_ids[rank] = flat (token*K+k) 索引。
#
# 不需要 `.item()` host sync：sorted_token_ids/expert_ids 按安全上界
# `numel + E*(B-1)` / `ceildiv(M_pad_upper, B)` 预分配。下游 grouped-GEMM
# kernel 通过指针读 num_tokens_post_padded 并对越过真实边界的 tile 提前
# return，过大 buffer 只多启动几个被 mask 掉的 program。
# ---------------------------------------------------------------------------
@triton.jit
def _small_batch_align_kernel(
    topk_ids_ptr,                # int32 (numel,)
    sorted_token_ids_ptr,        # int32 (M_pad_upper,)
    expert_ids_ptr,              # int32 (max_blocks_upper,)
    num_tokens_post_padded_ptr,  # int32 (1,)
    numel,
    num_experts,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,       # next_power_of_2(numel), <= 1024
    BLOCK_E: tl.constexpr,       # next_power_of_2(num_experts), <= 64
    BLOCK_B: tl.constexpr,       # next_power_of_2(BLOCK_SIZE), usually 32
):
    """小 batch 单 launch align 路径。

    vLLM 小 batch CUDA kernel 的 Triton 版：一个 program 拥有整个
    (token,k) 数组，算全部专家计数、cumsum padded、写 sorted ids /
    expert ids / 哨兵。完全跳过大 batch 的四 launch 流水。
    """
    n_offs = tl.arange(0, BLOCK_N)
    n_mask = n_offs < numel
    expert_for_token = tl.load(topk_ids_ptr + n_offs, mask=n_mask, other=-1)

    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < num_experts

    # counts[e] = sum(topk_ids == e)。shape: (BLOCK_E,)
    matches = expert_for_token[None, :] == e_offs[:, None]
    counts = tl.sum(tl.where(matches & n_mask[None, :], 1, 0), axis=1)
    counts = tl.where(e_mask, counts, 0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    inclusive = tl.cumsum(padded, axis=0)
    base = inclusive - padded
    total = tl.sum(padded, axis=0)
    tl.store(num_tokens_post_padded_ptr, total)

    b_offs = tl.arange(0, BLOCK_B)
    for expert in range(0, BLOCK_E):
        if expert < num_experts:
            expert_mask = e_offs == expert
            cnt = tl.sum(tl.where(expert_mask, counts, 0), axis=0)
            pad = tl.sum(tl.where(expert_mask, padded, 0), axis=0)
            base_e = tl.sum(tl.where(expert_mask, base, 0), axis=0)

            # 写真实 token。整个小 batch 由一个 program 拥有，前缀 rank
            # 确定性，无需 atomic。
            is_expert = expert_for_token == expert
            prefix = tl.cumsum(tl.where(is_expert, 1, 0), axis=0)
            tl.store(
                sorted_token_ids_ptr + base_e + prefix - 1,
                n_offs.to(tl.int32),
                mask=n_mask & is_expert,
            )

            # 填哨兵补齐 [base+cnt, base+pad)。
            pad_len = pad - cnt
            tl.store(
                sorted_token_ids_ptr + base_e + cnt + b_offs,
                numel + tl.zeros((BLOCK_B,), dtype=tl.int32),
                mask=b_offs < pad_len,
            )

            # 填该专家的 expert_ids 块。
            nb = pad // BLOCK_SIZE
            tl.store(
                expert_ids_ptr + base_e // BLOCK_SIZE + b_offs,
                expert + tl.zeros((BLOCK_B,), dtype=tl.int32),
                mask=b_offs < nb,
            )


@triton.jit
def _count_tokens_kernel(
    topk_ids_ptr,                # int32 (numel,)
    counts_ptr,                  # int32 (num_experts,) zero-init
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(counts_ptr + e, 1, mask=mask)


@triton.jit
def _meta_kernel(
    counts_ptr,                  # int32 (E,)
    base_ptr,                    # int32 (E,) — exclusive prefix sum
    counter_ptr,                 # int32 (E,) — atomic scatter cursor; init = base
    padded_ptr,                  # int32 (E,) — ceil(counts/B)*B
    num_tokens_post_padded_ptr,  # int32 (1,)
    num_experts,
    BLOCK_SIZE: tl.constexpr,    # token block size (== block_m)
    BLOCK_E: tl.constexpr,       # next_power_of_2(num_experts)
):
    offs = tl.arange(0, BLOCK_E)
    mask = offs < num_experts
    counts = tl.load(counts_ptr + offs, mask=mask, other=0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    inclusive = tl.cumsum(padded, axis=0)
    base = inclusive - padded
    tl.store(base_ptr + offs, base, mask=mask)
    tl.store(counter_ptr + offs, base, mask=mask)
    tl.store(padded_ptr + offs, padded, mask=mask)
    total = tl.sum(padded, axis=0)
    # 只在 0 号 lane 写 total。
    tl.store(num_tokens_post_padded_ptr + offs, total, mask=offs == 0)


@triton.jit
def _fill_kernel(
    base_ptr, counts_ptr, padded_ptr,
    expert_ids_ptr,
    sorted_token_ids_ptr,
    num_experts,
    sentinel,
    BLOCK_SIZE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    e = tl.program_id(0)
    chunk_id = tl.program_id(1)
    if e >= num_experts:
        return
    base_e = tl.load(base_ptr + e)
    cnt_e = tl.load(counts_ptr + e)
    pad_e = tl.load(padded_ptr + e)
    nb_e = pad_e // BLOCK_SIZE
    block_start = base_e // BLOCK_SIZE

    # 填 expert_ids[block_start : block_start + nb_e] = e。
    eoffs = chunk_id * CHUNK + tl.arange(0, CHUNK)
    e_int32 = e.to(tl.int32)
    tl.store(
        expert_ids_ptr + block_start + eoffs,
        e_int32 + tl.zeros((CHUNK,), dtype=tl.int32),
        mask=eoffs < nb_e,
    )

    # 哨兵补齐：每专家补齐长度 < BLOCK_SIZE（pad_e - cnt_e ∈ [0, BLOCK_SIZE)），
    # 单个 program 足够。
    if chunk_id == 0:
        soffs = tl.arange(0, BLOCK_SIZE)
        pad_len = pad_e - cnt_e
        smask = soffs < pad_len
        sentinel_v = sentinel.to(tl.int32)
        tl.store(
            sorted_token_ids_ptr + base_e + cnt_e + soffs,
            sentinel_v + tl.zeros((BLOCK_SIZE,), dtype=tl.int32),
            mask=smask,
        )


@triton.jit
def _scatter_tokens_kernel(
    topk_ids_ptr,                # int32 (numel,)
    counter_ptr,                 # int32 (num_experts,) — exclusive base offsets, mutated
    sorted_token_ids_ptr,        # int32 (M_pad_upper,)
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    e = tl.load(topk_ids_ptr + offs, mask=mask, other=0)
    # tl.atomic_add 返回旧值 -> 我们在专家桶内的 rank。
    rank = tl.atomic_add(counter_ptr + e, 1, mask=mask)
    tl.store(sorted_token_ids_ptr + rank, offs.to(tl.int32), mask=mask)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """:func:`moe_align_block_size_torch` 的 Triton 优化版。

    返回按安全上界（``numel + E*(B-1)``）预分配的过大 buffer，因此无需
    host 侧 ``.item()`` sync。下游 grouped-GEMM kernel 通过指针读
    ``num_tokens_post_padded`` 并跳过越过真实边界的 tile。

    每个专家桶内的输出顺序不确定（atomic-scatter），但每桶 flat token id
    的**多重集**以及 ``expert_ids`` / ``num_tokens_post_padded`` 与
    torch 参考实现完全一致。
    """
    T, K = topk_ids.shape
    numel = T * K
    device = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int32).contiguous()

    # 上界——保证无需 host sync 即可确定输出尺寸。
    M_pad_upper = numel + num_experts * (block_size - 1)
    max_blocks_upper = (M_pad_upper + block_size - 1) // block_size
    max_nb_per_expert = (numel + block_size - 1) // block_size + 1

    sorted_token_ids = torch.empty(
        M_pad_upper, dtype=torch.int32, device=device
    )
    # expert_ids 零初始化，使尾部（越过 num_tokens_post_padded）的槽位
    # 持有合法的 in-range 专家 id（0）。这很重要：调用方可能在切片前做
    # `expert_map[expert_ids]`，越界值会触发非法 gather。下游 GEMM kernel
    # 仍对越过 num_tokens_post_padded 的 tile 提前退出，所以该值其它情况
    # 不会被用到。
    expert_ids = torch.zeros(
        max_blocks_upper, dtype=torch.int32, device=device
    )
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)

    # vLLM 风格小 batch 快速路径：单 program 完成 count + cumsum + fill +
    # scatter，跳过大 batch 的四 launch 流水。
    if numel <= 1024 and num_experts <= 64:
        block_n = max(triton.next_power_of_2(numel), 2)
        block_e = max(triton.next_power_of_2(num_experts), 2)
        block_b = max(triton.next_power_of_2(block_size), 2)
        _small_batch_align_kernel[(1,)](
            flat, sorted_token_ids, expert_ids, num_tokens_post_padded,
            numel, num_experts,
            BLOCK_SIZE=block_size,
            BLOCK_N=block_n,
            BLOCK_E=block_e,
            BLOCK_B=block_b,
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    base = torch.empty(num_experts, dtype=torch.int32, device=device)
    counter = torch.empty(num_experts, dtype=torch.int32, device=device)
    padded = torch.empty(num_experts, dtype=torch.int32, device=device)

    # Stage A: 计数。
    grid_count = (triton.cdiv(numel, block),)
    _count_tokens_kernel[grid_count](flat, counts, numel, BLOCK=block)

    # Stage B: meta（单 program，BLOCK_E 条 lane）。
    block_e = max(triton.next_power_of_2(num_experts), 2)
    _meta_kernel[(1,)](
        counts, base, counter, padded, num_tokens_post_padded,
        num_experts,
        BLOCK_SIZE=block_size, BLOCK_E=block_e,
    )

    # Stage C: 填 expert_ids + 哨兵补齐。
    chunk = 64
    num_chunks = (max_nb_per_expert + chunk - 1) // chunk
    _fill_kernel[(num_experts, num_chunks)](
        base, counts, padded, expert_ids, sorted_token_ids,
        num_experts, numel,                     # 哨兵值 = numel
        BLOCK_SIZE=block_size, CHUNK=chunk,
    )

    # Stage D: scatter。
    _scatter_tokens_kernel[grid_count](
        flat, counter, sorted_token_ids, numel, BLOCK=block,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ---------------------------------------------------------------------------
# Triton kernel: BF16/FP16/FP32 grouped GEMM，无量化、无 bias。
# ---------------------------------------------------------------------------
@triton.jit
def _fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr,                  # MUL_ROUTED_WEIGHT == False 时可为空指针
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # 纯补齐行（无真实 token）—— 该 C tile 输出 0。
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + ((offs_token[:, None] // top_k) * stride_am
                      + offs_k[None, :] * stride_ak)
    b_ptrs = (b_ptr + off_experts * stride_be
              + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_grouped_gemm(
    A: torch.Tensor,                  # (T_a, K_in)
    B: torch.Tensor,                  # (E, N_out, K_in)
    C: torch.Tensor,                  # (T_a*K_topk, N_out)
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    mul_routed_weight: bool,
    top_k: int,
    block_m: int = 32, block_n: int = 64, block_k: int = 32, group_m: int = 8,
):
    EM = sorted_token_ids.numel()
    N = B.size(1)
    K = B.size(2)
    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)

    if A.dtype == torch.float16:
        compute_type = tl.float16
    elif A.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    else:
        compute_type = tl.float32

    _fused_moe_kernel[grid](
        A, B, C,
        topk_weights if topk_weights is not None else A,  # MUL_ROUTED_WEIGHT == False 时用 dummy
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
    )


def _silu_mul(gate_up: torch.Tensor, intermediate_size_local: int) -> torch.Tensor:
    """SiLU(x[:I]) * x[I:]。(M, 2I) -> (M, I)。"""
    gate, up = gate_up.split(intermediate_size_local, dim=-1)
    return torch.nn.functional.silu(gate) * up


def fused_experts_triton(
    hidden_states: torch.Tensor,        # (T, H)
    w13: torch.Tensor,                  # (E, 2*I_local, H)
    w2: torch.Tensor,                   # (E, H, I_local)
    topk_ids: torch.Tensor,             # (T, K)
    topk_weights: torch.Tensor,         # (T, K)
    expert_map: torch.Tensor | None = None,  # (E,) int32: -1 表非本地，否则专家 id
    block_m: int = 32,
    cache1: torch.Tensor | None = None, # 可选预分配 (>=T*K, 2*I_local)
    cache2: torch.Tensor | None = None, # 可选预分配 (>=T*K, I_local)
    cache3: torch.Tensor | None = None, # 可选预分配 (>=T*K, H)
    align_impl: str = "triton",         # "triton" 或 "torch"
) -> torch.Tensor:
    """Triton-fused MoE 前向。返回 (T, H)。

    Token 在调用方 rank 上**完整复制**；``expert_map`` mask 掉非本地专家
    （其 tile 输出 0）。跨 rank all-reduce 由调用方负责。

    三个中间 buffer 可由调用方提供以避免每步分配。kernel 输出的补齐行被
    ``token_mask`` mask 掉、且后续 reduce view 恰好 ``T*K``，不会被读到，
    因此上一次调用的残留值无害。
    """
    T, H = hidden_states.shape
    K = topk_ids.size(1)
    E, two_I, _ = w13.shape
    I_local = two_I // 2
    # ``topk_ids`` 在*全局*专家空间（[0, E_global)）。EP 时 ``w13.shape[0]``
    # 只是本地专家数，会错误地确定 align 桶大小——路由到非本地专家的 token
    # 会被静默丢弃（小 batch 路径）或触发越界 atomic add（大 batch 路径）。
    # 因此从 ``expert_map`` 推导全局 E（其长度恒为 E_global）。
    num_experts = expert_map.numel() if expert_map is not None else E
    TK = T * K

    def _take(buf, rows, cols, dtype):
        if buf is None:
            return torch.empty(rows, cols, dtype=dtype, device=hidden_states.device)
        if buf.size(0) < rows or buf.size(1) < cols:
            raise ValueError(
                f"preallocated cache too small: got {tuple(buf.shape)}, "
                f"need at least ({rows}, {cols})"
            )
        return buf[:rows, :cols]

    # 1. align tokens。
    align_fn = (
        moe_align_block_size_triton if align_impl == "triton"
        else moe_align_block_size_torch
    )
    sorted_ids, expert_ids, M_pad_t = align_fn(
        topk_ids, block_m, num_experts
    )
    if expert_map is not None:
        expert_ids = expert_map[expert_ids.to(torch.int64)].to(torch.int32)

    # 2. 第一段 GEMM：A=hidden_states, B=w13 -> cache1 (T*K, 2*I_local)
    c1 = _take(cache1, TK, two_I, hidden_states.dtype)
    _invoke_grouped_gemm(
        hidden_states, w13, c1,
        topk_weights=None,
        sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        num_tokens_post_padded=M_pad_t, num_valid_tokens=TK,
        mul_routed_weight=False, top_k=K, block_m=block_m,
    )

    # 3. SiLU·mul（eager torch——廉价 pointwise）。若调用方提供 buffer 则写入 c2。
    gate, up = c1.split(I_local, dim=-1)
    if cache2 is not None:
        c2 = _take(cache2, TK, I_local, hidden_states.dtype)
        c2.copy_(torch.nn.functional.silu(gate) * up)
    else:
        c2 = torch.nn.functional.silu(gate) * up

    # 4. 第二段 GEMM：A=c2（已在 (token,k) 行序），B=w2 -> c3 (T*K, H)。
    #    传 top_k=1 使 kernel 直接用 offs_token 作为 c2 行索引。
    c3 = _take(cache3, TK, H, hidden_states.dtype)
    _invoke_grouped_gemm(
        c2, w2, c3,
        topk_weights=topk_weights.reshape(-1).to(hidden_states.dtype),
        sorted_token_ids=sorted_ids, expert_ids=expert_ids,
        num_tokens_post_padded=M_pad_t, num_valid_tokens=TK,
        mul_routed_weight=True, top_k=1, block_m=block_m,
    )

    # 5. 在 k 上 reduce。
    return c3.view(T, K, H).sum(dim=1)
