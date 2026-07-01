# 本轮改动文档索引

两笔提交的详细修改、原理与排查过程：

| 文档 | 对应提交 | 内容 |
|---|---|---|
| [Logger_Loader_MoE_与采样器乱码修复.md](./Logger_Loader_MoE_与采样器乱码修复.md) | `c684ced` | vLLM 风格 logger、多线程权重加载、Qwen3-MoE 支持（TP/EP 保留/去量化）、推理崩溃上下文日志、greedy 采样乱码修复（含逐算子对比 HF 的排查过程）、attention backend 别名修复 |
| [Overlap_CudaGraph_竞态修复.md](./Overlap_CudaGraph_竞态修复.md) | `8c064dc` | overlap + CUDA-graph 流竞态：现象、四象限二分定位、同步二分、根因、修复方案选型（含吞吐对比表）、`wait_stream` 非阻塞屏障、验证 |

## 快速结论

- **乱码根因**：`flashinfer.sampling.softmax(temp=0)` 返回均匀分布而非 one-hot，且 `top_k_top_p_sampling_from_probs(top_k=0)` 丢弃所有 token → greedy 永远采样到 0。改用 PyTorch softmax + Gumbel-max 路径修复。
- **forward 正确性**：embedding 精确、各层 fp16 噪声 <1%、末层 logits top-5 与 HF 完全一致。
- **竞态根因**：overlap 下 schedule_stream（step N+1 准备）的 GPU 活动与 forward_stream（step N graph replay）并行，偶发 stale 读 → token 翻转。`schedule_stream.wait_stream(forward_stream)`（仅 graph 模式、非阻塞）修复，774 tok/s（racy 838 的 92%，且优于其它正确方案）。
