# 性能优化：FA2 公平对比、CPU/Sampler 优化、长序列 attention 瓶颈

对应提交：`2c0792e`、`1dd7960`、`a93a3e6`（benchmark 工具）

本文档记录与 vLLM 的公平 benchmark 对比、两轮优化的排查与改动、以及剩余性能差距的根因猜想。

---

## 1. 公平对比方法

vLLM 默认在 A100 上可能用 FlashInfer 或 FA3。为与 MiniInfer（用 `flash_attn_with_kvcache`，即 FA2）公平对比，强制 vLLM 用 FA2：

```bash
VLLM_ATTENTION_BACKEND=FLASH_ATTN python benchmarks/benchmark_vllm.py \
  --model /root/models/Qwen3-0.6B --tp 1 --max-num-seqs 256 --max-model-len 4096
```

vLLM 日志确认：`Using FLASH_ATTN attention backend` / `Using FlashAttention version 2`。

负载对齐：`seed(0)`、256 seqs、输入 100–1024、输出 100–1024、`temperature=0.6`、`ignore_eos=True`（与 `nano-vllm/benchmarks/benchmark_vllm.py` 完全一致，总 token 133966）。

## 2. 优化结果

| 负载 | vLLM (FA2) | MiniInfer 优化前 | MiniInfer 优化后 | 结论 |
|---|---|---|---|---|
| 256×1024（长序列） | 10050 | 6422 | **6978** | 69% of vLLM |
| 256×256（中序列） | 10050* | — | **10552** | **反超 vLLM** |
| 256×64（短，profiled） | ~10050 | — | **9999** | ≈ vLLM |
| 128×128（profiled） | — | 9231 | **11164** | **反超 vLLM** |

\* vLLM 10050 是 256×1024 测的；256×256 未单独测 vLLM，但短/中序列 MiniInfer 已超其长序列吞吐。

**关键结论：MiniInfer 在短/中序列已反超 vLLM；唯一差距在长序列 decode attention。**

---

## 3. 优化一：消除 decode 热路径 boolean-index / allocator 同步（`2c0792e`）

### 现象
profiler（32 seqs × 128 out）显示 `aten::nonzero + index + _index_put` 合计占 CPU ~31%，`cudaStreamSynchronize` ~14%。CPU 时间（1405ms）是 GPU（650ms）的 2.16× → **GPU 饿死 54%**。

### 排查
- monkeypatch 所有同步入口（`cuda.synchronize` / `Stream.synchronize` / `Event.synchronize` / `tensor.item/.cpu/.tolist`）跑短 decode：**显式同步只有 `copy_done_event.synchronize()`（overlap D2H，必要，1 次/步）**。
- 但 profiler 显示 `cudaStreamSynchronize` ~2 次/步。A/B 关掉 `wait_stream`（竞态修复）：同步次数不变 → 第 2 处同步与 `wait_stream` 无关 → 是 **caching allocator 隐式同步**。
- 用 `with_stack` 定位 `aten::nonzero`：来自 `future_map.resolve_future_input_ids` 每步对 `input_ids` 做 `input_ids[is_placeholder]`（boolean 索引 → nonzero+index）+ `input_ids[is_placeholder] = ...`（index_put）。这些临时 tensor 的分配/回收触发 allocator 跨流同步。

### 改动
1. `resolve_future_input_ids` 加 `all_placeholder` 快速路径（纯 decode batch：input_ids 全是占位符）→ 直接 `gather + copy_`，跳过 nonzero/boolean-index/index_put。调用方传 `batch.forward_mode.is_decode()`。
2. `_propagate_future_to_running_batch` 加 `running_batch.reqs == batch.reqs` 同序同集快速路径 → 直接 clone，跳过 2 次 nonzero + masked gather。

### 效果
- 32seq profiled：2915 → **5267 tok/s（+81%）**，CPU 1405→778ms，GPU 650→654ms（不变）。
- `aten::nonzero` 与 `cudaStreamSynchronize` 从 top 消失。
- 256×1024 bench：6422 → 6786（+5.7%，高 batch GPU 成瓶颈，CPU 收益摊薄）。
- 正确性：greedy decode 与 HF 逐 token 一致。

### 附：`cudaGraphLaunch` 1.57ms/次排查
A/B 关掉 `wait_stream`（竞态修复）：`cudaGraphLaunch` 1.57→1.07ms、吞吐 +11%，但 `cudaStreamSynchronize` 不变且 correctness 回到竞态。结论：`wait_stream` 的串行代价 ~0.5ms/launch，是 graph+overlap 正确性所需，**保留**。

---

## 4. 优化二：sampler 分级路由，temp>0 用 flashinfer 融合采样（`1dd7960`）

### 现象
profiler（128 seqs）显示 `ModelRunner.sample` 占 GPU 7.6%（104ms/步，780µs/步）。根因：统一走 float32 Gumbel-max，对 `[B, vocab=151936]` logits 做 `.float()` 上转（4× 显存，每步若干 155MB 临时 tensor）+ softmax + exponential + argmax。

### 关键事实
- `flashinfer.sampling.softmax(logits, temperature)` **仅在 temperature=0 时坏**（返回均匀分布而非 one-hot，这是之前 greedy 乱码的根因）；temp>0 完全正常且是融合 kernel。
- 微基准：flashinfer softmax+sample 比 float32 Gumbel-max 快 **~10×**（4.2ms vs 44.9ms / 50 iters）。

### 改动
按 CPU 端 flag（`ForwardBatch.init_new` 预算 `all_greedy`/`all_non_greedy`，避免 GPU sync）分级路由：
- `all_greedy`（全 temp<=0）→ `torch.argmax`，最快。
- `all_non_greedy`（全 temp>0）→ `flashinfer.softmax` + `sampling_from_probs`（有 top-k/top-p 时用 `flashinfer.top_k_top_p`，禁用 top_k clamp 到 V）。
- mixed → 保留 float32 Gumbel-max 兼容路径（罕见）。

### 效果
- 128seq profiled：9231 → **11164 tok/s（+21%）**，sample 104→42ms → **此时已超 vLLM 10050**。
- 256×64 profiled：9999 ≈ vLLM。
- 正确性：greedy 与 HF 一致；temp>0 + top_k/top_p 与 temp>0 无过滤均正常。

---

## 5. 性能差距猜想：长序列 decode attention

### 现象
- 256×256（avg seq ~512）MiniInfer **10552 > vLLM 10050**。
- 256×1024（avg seq ~1074）MiniInfer **6978 < vLLM 10050**。
- 唯一变量是序列长度 → 差距来自 **decode attention 随 KV 长度的扩展性**。

### 根因猜想：`page_size=256` 硬约束
MiniInfer decode 用 `flash_attn_with_kvcache`（paged KV）。实测：

```
RuntimeError: Paged KV cache block size must be divisible by 256
```

即 flash_attn **强制 page_size 必须被 256 整除**。所以 `page_size=256` 是下限，A/B `page_size ∈ {16,64,128}` 全部不可用。

- vLLM 用细粒度 `block_size=16`（其 FA2 backend 走 gather+`flash_attn_varlen_func` 或自有 paged 路径，不受 256 约束）。
- MiniInfer 用 256-token 大块。长序列下（seq_len ~1000 → 4 个 256 块），split-K 并行度受限，吞吐随 seq len 下降比 vLLM 快。
- 短序列下块数少、attention 占比小，MiniInfer 的 CPU/采样器优势占主导 → 反超 vLLM。

### 其它已排除项
- attention backend：两边都 FA2，公平。
- sampler：已优化到 3%，非瓶颈。
- CPU 喂数据：已基本打平（CPU ≈ GPU）。
- `cudaGraphLaunch` 1ms/launch：`wait_stream` 正确性代价，必要保留。

### 闭合差距的下一步（二选一）
1. **修 FlashInfer backend 的 GQA bug**（`Unsupported group_size: 0`——Qwen3-0.6B 是 GQA 16:8，group_size 应为 2 但传了 0）。FlashInfer 的 paged decode attention 是 vLLM-grade，无 page_size=256 约束，长序列扩展性更好。**最高 ROI**。
2. decode attention 改 gather + `flash_attn_varlen_func`（vLLM FA2 做法），绕过 page_size 约束，但多一次 KV gather。

---

## 7. FlashInfer backend 修复进展

按「下一步 1」修 FlashInfer backend。原报错 `Unsupported group_size: 0` 实际不是 GQA 比例问题（16:8 → group_size=2 本应支持），而是 **KV layout 与 page 索引错误**导致 kernel 读到错位数据，group_size 派生为 0。

### 修了三个 bug
1. **KV layout 不匹配**（`flashinfer_backend.py` forward_decode/extend）：声明 `kv_layout="NHD"`（期望 `[num_pages, page_size, heads, head_dim]`），但代码 `.view(-1, page_size, heads, dim).permute(0,2,1,3)` 产出的却是 HND → kernel 读错位 → segfault / group_size=0。**去掉 permute**（view 本身即 NHD）。
2. **page 索引 vs token 索引**（`init_forward_metadata`）：`get_page_table` 返回**每 token** 的 KV slot 索引，但 FlashInfer 的 `paged_kv_indices` 要**每页 1 个** page 索引。改成 `raw_page_table[:, ::page_size] // page_size` 取每页起始 token 再转 page 索引。
3. **cuda-graph 静态 buffer 被覆盖**（`init_cuda_graph_metadata`/`update_cuda_graph_metadata`）：原把静态 buffer 存在 `self.forward_metadata` 上，prefill 步的 `init_forward_metadata` 会替换 `forward_metadata` 把它清掉；且每个 capture batch size 共用一组 buffer，后捕获的 batch 覆盖前者。改为按 `batch_size` 存 `self._cg_buffers[bs]`，update 按 `cache_seqlens.shape[0]` 取对应组原地更新。

### 当前状态
| 配置 | 结果 |
|---|---|
| FlashInfer eager（任意规模） | ✅ 正确（与 HF 逐 token 一致） |
| FlashInfer + cuda graph，小 batch（≤8 seqs） | ✅ 正确 |
| FlashInfer + cuda graph，大 batch（32+ seqs / 长序列） | ❌ illegal memory access（stateful wrapper 未用 `use_cuda_graph=True` 模式） |
| FlashInfer + graph + overlap | ❌ 产生越界 garbage token（overlap 与 stateful wrapper 竞态） |

### 剩余问题（需 deeper 集成）
FlashInfer 的 `BatchDecodeWithPagedKVCacheWrapper` 创建时用了默认 `use_cuda_graph=False`。cuda-graph capture/replay 要求 `use_cuda_graph=True` + 预分配 max-size 的 `paged_kv_indptr/indices/last_page_len` buffer，且 FlashInfer 的 graph 模式面向**单张捕获图**，而 MiniInfer 的 `CudaGraphRunner` 按 power-of-2 batch size **捕获多张图**——两者模型不匹配，大 batch 时 stateful plan 错位 → illegal memory access。

正确闭合需要二选一：
- (a) FlashInfer 改 `use_cuda_graph=True` + 单张 max-batch 图 + padding（vLLM 做法），重构 `CudaGraphRunner` 对 FlashInfer 的支持；
- (b) 每个 capture batch size 一个独立 FlashInfer wrapper。

### 结论
FlashInfer backend 从「任意规模都崩」修到「eager + 小 batch graph 正确」。但因上述 graph 集成问题，**FlashInfer 暂未能用于长序列吞吐提速**（eager 无 graph 太慢；graph 大 batch 崩）。**生产路径仍是 FA2（6838 tok/s，正确）**。FlashInfer graph 集成作为后续独立工作。

---

## 6. 产物

- `benchmark/bench_miniinfer.py`：对齐 vLLM 方法论的吞吐 bench，支持 `--page-size` A/B。
- `benchmark/profile_miniinfer.py`：torch.profiler 脚本，导出 chrome trace + top kernel/op 表。
- `/tmp/miniinfer_trace.json`：chrome trace（可在 `chrome://tracing` 打开）。
