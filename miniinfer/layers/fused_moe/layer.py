"""FusedMoE 层（router top-k + experts + TP/EP all-reduce）。

并行模型
--------
- 进入本层的 token 在全局 TP 组的每个 rank 上*完整复制*。（attention 切
  head 但通过 RowParallelLinear 的 all-reduce 还原回完整 hidden。）
- EP：每个 rank 拥有连续的一段专家；其余靠 ``expert_map`` mask。输出在
  EP 组上求和。
- TP（MoE 内）：每个 rank 拥有每个本地专家 intermediate 维的连续切片。
  输出在 MoE-TP 组上求和。
- ``parallel.tp_group`` 是 *MoE*-TP 组（size = world_size / ep_size）；
  ``parallel.ep_group`` 是 EP 组。size=1 时均为 ``None``。

权重加载：per-expert HF 权重（``mlp.experts.{e}.{gate|up|down}_proj.weight``）
由 MoE 模型的 ``load_weights`` 调用 :meth:`FusedMoE.load_expert_weight` 路由进来。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from miniinfer.layers.fused_moe.dispatcher import (
    MoEDispatcher,
    MoEParallelConfig,
    prepare_mlp,
)
from miniinfer.layers.fused_moe.experts import FusedExperts


class FusedMoE(nn.Module):
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        norm_topk_prob: bool = True,
        parallel: MoEParallelConfig | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.parallel = parallel or MoEParallelConfig()

        # router gate 是 replicated（vLLM 约定）且不量化。
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)
        self.gate.weight.requires_grad = False

        self.experts = FusedExperts(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=self.parallel.tp_size,
            ep_size=self.parallel.ep_size,
            ep_rank=self.parallel.ep_rank,
            dtype=dtype,
            device=device,
        )
        self.dispatcher = MoEDispatcher(self.parallel, num_experts)

    # ------------------------------------------------------------------
    def _route(self, hidden: torch.Tensor):
        router_logits = self.gate(hidden)
        weights = F.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_ids, topk_weights.to(hidden.dtype), router_logits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, orig_shape[-1])

        # 1) prepare_mlp：需要时在 DP/CP 组上 all-gather token。
        full_hidden, scatter_idx = prepare_mlp(hidden_states, self.parallel)

        # 2) 在完整 token 集上跑 router。
        topk_ids, topk_weights, _ = self._route(full_hidden)

        # 3) 每个 rank 只算本地专家（FusedExperts 内靠 expert_map mask），
        #    相对 TP+EP 是 partial。
        partial = self.experts(full_hidden, topk_ids, topk_weights)

        # 4) 在 EP 与 MoE-TP 组上求和。
        full_out = self.dispatcher.combine(partial)

        # 5) 若上游被切分则 re-shard 回去。
        if scatter_idx is not None:
            out = full_out.index_select(0, scatter_idx)
        else:
            out = full_out

        if len(orig_shape) == 3:
            out = out.reshape(orig_shape)
        return out

    # ------------------------------------------------------------------
    # 权重加载 helper（MoE 模型加载 per-expert 权重时调用）。
    # ------------------------------------------------------------------
    def load_expert_weight(
        self,
        expert_id: int,
        which: str,
        weight: torch.Tensor,
        param_name: str = "weight",
    ):
        """把 per-expert HF 权重拷进堆叠参数。

        ``expert_id`` 是 checkpoint key 中的*全局*专家 id。``ep_size > 1``
        时只保留本地拥有的专家权重，其余静默跳过。
        ``which`` ∈ {"gate_proj", "up_proj", "down_proj"}。
        ``param_name`` ∈ {"weight", ...}——非量化路径只接受 ``"weight"``。

        TP 切分逻辑（与 nano-vllm 一致）：
          - gate_proj / up_proj：行切，取 ``[tp_rank*I_local, (tp_rank+1)*I_local)``，
            分别填入 w13 的前半（gate）与后半（up）。
          - down_proj：列切，取 intermediate 维的对应切片填入 w2。
        ``tp_size=1`` 时取整段。
        """
        if not (
            self.experts.local_expert_start
            <= expert_id
            < self.experts.local_expert_end
        ):
            return
        if param_name != "weight":
            raise ValueError(
                f"MiniInfer FusedMoE 不支持量化权重 ({param_name})，"
                f"仅接受 param_name='weight'"
            )
        local_id = expert_id - self.experts.local_expert_start
        I_local = self.experts.intermediate_size_local
        tp_rank = self.parallel.tp_rank
        if which == "gate_proj":
            shard = weight.narrow(0, tp_rank * I_local, I_local)
            self.experts.w13_weight.data[local_id, :I_local].copy_(shard)
        elif which == "up_proj":
            shard = weight.narrow(0, tp_rank * I_local, I_local)
            self.experts.w13_weight.data[local_id, I_local:].copy_(shard)
        elif which == "down_proj":
            shard = weight.narrow(1, tp_rank * I_local, I_local)
            self.experts.w2_weight.data[local_id].copy_(shard)
        else:
            raise ValueError(which)
