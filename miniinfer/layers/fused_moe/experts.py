"""逐专家 fused MLP（堆叠权重），可选 TP+EP 切分。

``ep_size > 1`` 时，只有本地拥有的专家在 ``w13_weight`` / ``w2_weight``
中存权重；远程专家靠 ``expert_map``（远程为 -1、本地为本地索引）mask 掉。

堆叠权重布局（每 rank，TP+EP 切分后）：
    w13_weight: (E_local, 2 * I_local, H)        # 行按 TP 切
    w2_weight:  (E_local, H,           I_local)  # 列按 TP 切

``TP > 1`` 时 ``forward`` 返回的 *partial* 输出需在 TP 组上求和；EP partial
同理需 EP all-reduce。``MoEDispatcher.combine`` 负责这两次 reduce。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from miniinfer.layers.fused_moe.fused_moe_kernel import fused_experts_triton


class FusedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int = 1,
        ep_size: int = 1,
        ep_rank: int = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert intermediate_size % tp_size == 0, (
            f"intermediate_size={intermediate_size} not divisible by tp_size={tp_size}"
        )
        assert num_experts % ep_size == 0, (
            f"num_experts={num_experts} not divisible by ep_size={ep_size}"
        )
        self.num_experts = num_experts
        self.num_local_experts = num_experts // ep_size
        self.local_expert_start = ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.intermediate_size_local = intermediate_size // tp_size
        kw = {"dtype": dtype, "device": device}
        # 仅 BF16/FP16/FP32，无量化：直接用浮点堆叠权重。
        self.w13_weight = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_size_local, hidden_size, **kw),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, self.intermediate_size_local, **kw),
            requires_grad=False,
        )

        # expert_map 是 (num_experts, ep_rank, ep_size) 的纯函数，构造后不变。
        # 注册为 buffer 以便只建一次（非每步 forward 分配）并随 .to()/.cuda()
        # 一起搬运。-1 标远程专家；本地专家持有其在 w13/w2 中的本地行索引。
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32, device=device)
        local_ids = torch.arange(self.num_local_experts, dtype=torch.int32, device=device)
        expert_map[self.local_expert_start : self.local_expert_end] = local_ids
        self.register_buffer("expert_map", expert_map, persistent=False)

        # 单块平坦 workspace 供 triton fused 路径使用。三个中间量被切成
        # *连续*的平坦子块（非列切片），保证各自连续（下游 .view 廉价）且
        # GEMM2 的输入/输出占据不相交内存：
        #
        #   flat[0          : TK*I    ] -> c2  (silu·mul out, GEMM2 input A)
        #   flat[TK*I       : TK*I+TK*H] -> c3 (GEMM2 output C, 连续)
        #   flat[0          : TK*2I   ] -> c1  (GEMM1 out; 与 c2 & c3 区域重叠，
        #                                       但 GEMM2 跑前已死——kernel 启动 /
        #                                       stream 顺序使其安全）
        #
        # 唯一的 kernel 内 hazard 是 GEMM2（A=c2, C=c3 被同一次 launch 的并发
        # 列 tile 触碰）；上面 c2 与 c3 不相交，故无读写竞争。c1 vs c2 重叠
        # 安全，因为 silu 在拷进 c2 前已完整算出结果。
        #
        # 只增不减；warmup_model() 按最大 TK 一次分配，故永不重分配、基址在
        # CUDA-graph replay 间稳定。
        self._workspace: torch.Tensor | None = None

    def _ensure_workspace(self, tk: int, device, dtype) -> torch.Tensor:
        I = self.intermediate_size_local
        H = self.hidden_size
        need = tk * max(2 * I, I + H)
        ws = self._workspace
        if (
            ws is None
            or ws.numel() < need
            or ws.device != device
            or ws.dtype != dtype
        ):
            ws = torch.empty(need, dtype=dtype, device=device)
            self._workspace = ws
        return ws

    def forward(
        self,
        hidden_states: torch.Tensor,        # (T, H)，完整 token 集
        topk_ids: torch.Tensor,             # (T, K)
        topk_weights: torch.Tensor,         # (T, K)
        expert_mask: torch.Tensor | None = None,  # legacy，忽略
    ) -> torch.Tensor:
        emap = self.expert_map
        if emap.device != hidden_states.device:
            emap = emap.to(hidden_states.device)
            self.expert_map = emap
        if hidden_states.is_cuda:
            I = self.intermediate_size_local
            H = self.hidden_size
            tk = hidden_states.size(0) * topk_ids.size(1)
            ws = self._ensure_workspace(tk, hidden_states.device, hidden_states.dtype)
            c1 = ws[: tk * 2 * I].view(tk, 2 * I)
            c2 = ws[: tk * I].view(tk, I)
            c3 = ws[tk * I : tk * I + tk * H].view(tk, H)
            return fused_experts_triton(
                hidden_states,
                self.w13_weight,
                self.w2_weight,
                topk_ids.to(torch.int32),
                topk_weights,
                expert_map=emap,
                cache1=c1,
                cache2=c2,
                cache3=c3,
            )
        # CPU eager fallback（gloo 多 rank parity 测试用）。
        return self._eager_forward(hidden_states, topk_ids, topk_weights)

    def _eager_forward(self, hidden_states, topk_ids, topk_weights):
        T, H = hidden_states.shape
        out = torch.zeros_like(hidden_states)
        E = self.num_experts
        oh = F.one_hot(topk_ids.long(), num_classes=E).permute(2, 1, 0)  # (E, K, T)
        local_present = oh.sum(dim=(-1, -2)) > 0
        # 只算本地拥有的专家。
        for e in range(self.local_expert_start, self.local_expert_end):
            if not local_present[e]:
                continue
            local_e = e - self.local_expert_start
            mat = oh[e]
            k_idx, t_idx = torch.where(mat)
            x = hidden_states.index_select(0, t_idx)
            w13 = self.w13_weight[local_e]
            gate_up = F.linear(x, w13)
            I_local = self.intermediate_size_local
            gate, up = gate_up.split(I_local, dim=-1)
            mid = F.silu(gate) * up
            y = F.linear(mid, self.w2_weight[local_e])
            y = y * topk_weights[t_idx, k_idx, None].to(y.dtype)
            out.index_add_(0, t_idx, y)
        return out
