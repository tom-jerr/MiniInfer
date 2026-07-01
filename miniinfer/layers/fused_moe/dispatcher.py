"""MoE 并行布局：TP / EP / TP+EP，token 全 replicated。

设计（镜像 vLLM ``FusedMoE``，去掉底层量化 kernel）：

- 进入 ``FusedMoE.forward`` 的 token 在 MoE 组的每个 rank 上都**完整
  复制**。若 attention 按 DP/CP 切了 token，上游的 ``prepare_mlp`` 会
  先做一次 all-gather。
- EP 纯靠 mask 实现：每个 rank 只算自己拥有的专家
  （``[ep_rank*E/EP, (ep_rank+1)*E/EP)``），partial 输出在 EP 组上
  all-reduce 求和。
- TP 切每个专家的 intermediate 维度（``w13`` 行并行、``w2`` 列并行），
  ``w2`` 之后在 TP 组上 all-reduce。
- TP 与 EP 正交：同时开启时，TP all-reduce（专家内）与 EP all-reduce
  （专家间）合成对合并组的一次 all-reduce。

本模块**不**自行创建 process group；调用方传入 torch.distributed 的
group（或 ``None`` 表示 size=1）。size=1 时所有 all-reduce / all-gather
都是 no-op，单 GPU 跑时不会触发任何 dist 调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class MoEParallelConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    tp_group: Optional["dist.ProcessGroup"] = None
    ep_group: Optional["dist.ProcessGroup"] = None
    # 若 attention 按 DP/CP 切了 token，给出需要 all-gather 的组。
    dp_group: Optional["dist.ProcessGroup"] = None
    dp_size: int = 1
    dp_rank: int = 0


class MoEDispatcher:
    """基于 mask 的 dispatcher（无 all-to-all）。

    每个 rank 已持有全部 token。``combine`` 对各 rank 的 partial 专家
    输出做跨 rank all-reduce。
    """

    def __init__(self, parallel: MoEParallelConfig, num_experts: int):
        assert num_experts % parallel.ep_size == 0, (
            f"num_experts={num_experts} must be divisible by ep_size={parallel.ep_size}"
        )
        self.parallel = parallel
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // parallel.ep_size
        self.local_expert_start = parallel.ep_rank * self.experts_per_rank
        self.local_expert_end = (parallel.ep_rank + 1) * self.experts_per_rank

    def combine(self, partial: torch.Tensor) -> torch.Tensor:
        """在 EP 与 TP 组上对 partial 专家输出求和。

        ``partial`` 是本 rank ``FusedExperts(...)`` 的输出（已按 topk
        权重加权、并在本地专家上 reduce 完）。TP 与 EP 组独立时需要两次
        all-reduce，顺序不影响求和结果。

        NOTE: 传 ``group=None`` 给 ``dist.all_reduce`` 会用默认（WORLD）
        组——当 ``ep_size==1`` 时它正是正确的 TP 组。因此**不要**用
        ``is not None`` 守卫，否则 ``tp>1, ep==1`` 时会漏掉 TP all-reduce
        导致 MoE 输出错乱。size==1 时下面两个分支都不进，自然 no-op。
        """
        if self.parallel.ep_size > 1:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.parallel.ep_group)
        if self.parallel.tp_size > 1:
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.parallel.tp_group)
        return partial


def prepare_mlp(
    hidden: torch.Tensor,
    parallel: MoEParallelConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """若 attention 把 token 按 DP/CP 切了，做一次 all-gather。

    返回 ``(full_hidden, scatter_idx)``。未 gather 时 ``scatter_idx`` 为
    None；否则它是把本 rank shard 映射回去的索引，供调用方在 MoE 后
    re-shard。
    """
    if parallel.dp_size <= 1 or parallel.dp_group is None:
        return hidden, None

    T_local, H = hidden.shape
    T_full = T_local * parallel.dp_size
    full = torch.empty(T_full, H, dtype=hidden.dtype, device=hidden.device)
    dist.all_gather_into_tensor(full, hidden.contiguous(), group=parallel.dp_group)
    # 本 rank 的 shard 在 [dp_rank*T_local : (dp_rank+1)*T_local)。
    start = parallel.dp_rank * T_local
    scatter_idx = torch.arange(start, start + T_local, device=hidden.device)
    return full, scatter_idx
