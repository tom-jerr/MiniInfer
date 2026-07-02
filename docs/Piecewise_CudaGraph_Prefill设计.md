# Piecewise CUDA Graph for Prefill（MoE-extensible）

对应提交：piecewise cuda graph 框架（`miniinfer/engine/piecewise_cuda_graph.py` + FA2 extend cuda-graph metadata）。

## 背景

prefill 走 varlen attention，`total_tokens` 随 batch 变化，无法像 decode 那样按 power-of-2 batch size 直接单图捕获。vLLM 的 cuDAG 用 **piecewise**：把 forward 拆成 segment，**静态形状的（GEMM/norm/embedding）用 cuda graph 捕获**，**动态形状的（varlen attention、MoE router/grouped-GEMM）eager**，二者通过预分配 buffer 串联。

## 设计（可扩展，MoE-ready）

### Segment 注册机制
```python
@dataclass
class Segment:
    name: str
    eager: bool          # True=动态形状每步 eager（varlen attn / MoE router）；False=按 bucket 捕获
    run: Callable[[ReplayContext], None]
```
`PiecewiseCudaGraphRunner.register_segment(Segment(...))` 按顺序注册，capture/replay 按顺序串联。MoE 接入时把 router+grouped-GEMM 注册为 eager segment，前后的 dense GEMM 仍 captured——**无需改 runner**。

### v1 dense（本提交）
- 按 token bucket（powers of 2 ≤ `max_extend_len`：1024/2048/4096/8192）捕获整层 forward。
- **dummy padding seq** 把 `total_tokens` 补到 bucket：dummy token 的 `out_cache_loc` 指向保留的 page 0（KV 写入 page 0，覆盖无害）；varlen attention 处理 `[actual + dummy] = [bucket]`，输出 `[bucket]`，末尾取 `[:actual]`。
- 静态 buffer（per bucket）：`input_ids/positions/out_cache_loc [bucket]`、`cu_seqlens_q/k [max_nseq+2]` 等，replay 前 in-place 更新。
- FA2 后端新增 `init_extend_cuda_graph_metadata` / `update_extend_cuda_graph_metadata`（per-bucket 静态 cu_seqlens，避免后捕获覆盖前者——与 decode 的同类 bug 同理）。

### MoE 扩展点（后续）
dense 的「整层 forward」segment 拆成：
- `pre_attn`（captured）：input_norm + qkv + qk_norm + rope。
- `attn`（eager）：varlen attention，写 o 进 `[bucket]` buffer。
- `post_attn`（captured）：o_proj + residual + post_norm + **dense MLP（captured）** 或 **MoE（eager: router + grouped-GEMM）**。

只需注册对应 segment，runner 按序串联。

## 当前状态

框架已就位，但 **v1 capture 默认关闭**（`MINIINFER_PREFILL_CG=1` 开启尝试，失败自动回落 eager prefill，无回归）。

### 待修（v1 capture 未跑通）
1. **capture-unsafe op**：capture 时报 `cudaErrorStreamCaptureInvalidated`。疑似 prefill forward 内有 capture 不安全的操作（如 `Qwen3ForCausalLM.forward` 的 last-token 索引 `torch.tensor(last_token_indices, device=...)` 的 H2D，或某个隐式 sync）。需用 `CUDA_LAUNCH_BLOCKING=1` 定位。
2. **last-token 索引与 graph 切片不匹配**：`Qwen3ForCausalLM.forward`（extend）对 logits 做 last-token 索引（输出 `[nseq]`），但 piecewise graph 需要 `[bucket]` 全量 logits 再 `[:actual]` 切片。需为 capture 提供一个「全量 logits」forward 路径（或 graph 内不索引、graph 外切片）。

### 已验证
- 框架导入、segment 注册、capture/replay 结构、FA2 extend metadata 方法均就位。
- 默认关闭时引擎正常（eager prefill，greedy 与 HF 一致，无回归）。
- 命中 prefix cache 或超 bucket 时自动回落 eager（`replay` 返回 `None`）。

## 文件
- `miniinfer/engine/piecewise_cuda_graph.py`：`Segment` / `ReplayContext` / `PiecewiseCudaGraphRunner`。
- `miniinfer/layers/attention_backend/flashattention_backend.py`：FA2 `init_extend_cuda_graph_metadata` / `update_extend_cuda_graph_metadata`。
- `miniinfer/engine/model_runner.py`：`forward_extend` 优先用 piecewise graph（`MINIINFER_PREFILL_CG=1` 时），`init_cuda_graph` 末尾初始化 runner。

## 性能预期
prefill 的 gap（MiniInfer 88k vs vLLM 134k in_tok/s）主因是 GEMM-bound + eager launch 开销。piecewise cuda graph 消除 launch 开销（profile 显示 `cudaLaunchKernel` 8572 次/prefill），预计 prefill 提速 ~1.2×（GEMM 本身的 cuBLAS 效率差距仍需单独优化）。闭合 v1 capture 后可验证。
