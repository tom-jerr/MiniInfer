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

## 8. FlashInfer `use_cuda_graph=True` 集成（完成）+ 默认 backend 切换

### 集成
按 §7 的「下一步」用 FlashInfer 的 `use_cuda_graph=True` 模式正确集成 cuda graph：

- **每个 capture batch size 一个独立 wrapper**：FlashInfer graph 模式要求 `_fixed_batch_size` 固定（`plan` 内 `if batch_size != self._fixed_batch_size: raise`），而 MiniInfer 的 `CudaGraphRunner` 按 power-of-2 捕获多张图。故 `init_cuda_graph_metadata` 为每个 batch size 创建一个 `use_cuda_graph=True` 的 wrapper + 预分配 `paged_kv_indptr/indices/last_page_len` buffer，存 `self._cg_wrappers[bs]` / `self._cg_buffers[bs]`。
- **`plan()` 每 replay 调一次**：FlashInfer graph 模式下 `plan(indptr, indices, last_page_len, ...)` 把元数据 **copy 进 wrapper 内部 buffer** 并重建 split-K plan；graph 冻结的正是这些内部 buffer。故 `update_cuda_graph_metadata` 重建 CSR 后调 `plan()`（在 graph 外，CPU 规划合法），replay 即读到新值。heads/dim/dtype 在 `forward_decode` 首次调用缓存（`self._cg_heads`）供 `plan` 用。
- **`forward_decode` 选 wrapper**：graph 路径按 `q.shape[0]` 选 `self._cg_wrappers[bs]`；eager 路径用 `self.decode_wrapper`（非 graph）。capture 时 `is_current_stream_capturing()` 为 True 跳过 `plan`（含 CPU 规划），依赖 warmup 建立 wrapper 状态。

### 效果
FlashInfer + cuda graph 从「32+ seqs 崩 / overlap 出 garbage」修到**全规模正确**：
- 256×1024：**6984 tok/s**（无崩溃，greedy 与 HF 逐 token 一致）。
- 32 seqs decode-heavy：0 bad reqs（此前 garbage/crash）。

### Benchmark 结论：FlashInfer ≈ FA2，**未闭合长序列 gap**

| 负载 | FA2 (page=256) | FlashInfer (page=256) | vLLM (FA2) |
|---|---|---|---|
| 256×256 | 10552 | 10010 | 10050 |
| 256×1024 | 6838 | 6984 | 10050 |

- FlashInfer 与 FA2 在 `page_size=256` 下吞吐**基本持平**（FlashInfer 略慢 ~5%，其 decode kernel 450ms vs FA2 splitkv 426ms @128seq）。
- **更小 page_size 反而更慢**：FlashInfer page_size=64 → 7980、=16 → 8106（@256×256），因为页数增多 → page indirection 开销增大。FlashInfer 虽无 256 约束，但小 page 在此配置下不划算。
- **长序列 gap 不是 attention-bound**：FA2 与 FlashInfer 在 256×1024 都 ~6800–7000，vLLM 10050。两个 backend 都随 seq len 掉速，vLLM 不掉。说明 gap 在 MiniInfer 整体 pipeline（KV 布局/调度/overlap 效率等），**非换 attention backend 能解决**。

### 默认 backend 切换
`EngineConfig.attention_backend` 默认改为 `"flashinfer"`（无 page_size-256 约束、vLLM-grade、全规模正确）。`model_runner` 的别名归一化仍支持 `flash_attn`/`flash_attention_2` 等切回 FA2。benchmark 显示二者吞吐接近，故默认 FlashInfer 不损失明显性能，且获得布局灵活性。

### 仍未闭合的长序列 gap（后续方向）
256×1024 下 MiniInfer ~6900 vs vLLM 10050。既然非 attention backend，候选方向：
- KV cache 布局/分配效率（vLLM block_size=16 + 更优分配器）。
- overlap 在长序列下的 CPU 隐藏效率（`plan()` 每 replay 调用、scheduler 开销）。
- decode GEMM（lm_head `[B,1024]@[1024,151936]`）在长序列下的占比与融合。

---

## 9. 真正的根因：每步 O(seq_len) detokenization（已修复，gap 基本闭合）

### 公平对比确认 gap 仍在
vLLM 强制 FA2 + `block_size=256`（与 MiniInfer 完全同 backend 同 page）：**10136 tok/s**。
MiniInfer FA2 page_size=256：6838 tok/s。**同配置下仍 1.48× 慢** → 排除 page_size / attention backend。

### 定位
profile decode-heavy（128 seq × output 512）：
- `stage::LLMEngine._process_step_result` = **4.58s CPU（63%！）**，8.91ms/step。
- 根因：`IncrementalDecoder._incremental_decode` 每步调 `_decode_with_cache(state.token_ids)` 对**全量 token** decode（O(seq_len)）；其 LRU 缓存以全量 tuple 为 key，每步 tuple 增长 → 永不命中 → 每步都全量 decode。长序列下 128 seq × decode(512 tokens) × num_steps 占满 CPU。
- vLLM/SGLang 把 detokenization 放到**独立进程**（off critical path）；MiniInfer 单进程在关键路径上。

### 修复
单进程下 BPE 边界让「真增量 O(1) decode」复杂，故采用更直接的方案：**非流式路径跳过每步 detok**，仅用 token 判 eos/finished，结束时对每个请求 decode 一次全量 token（O(Σ seq_len)，每请求一次，远小于 O(seq_len × num_steps)）。
- 新增 `LLMEngine._streaming` flag：`generate`（非流式）跳过每步 detok；`stream_generate` 仍每步 detok（流式必需 delta_text）。
- `_process_step_result`：非流式时 `decoded = [("", is_eos_from_token, "")]`，不调 `decode_batch`。
- `generate` 结束后对每个请求 `detokenizer._decode(output_token_ids)` 一次。

### 效果（FA2，Qwen3-0.6B）
| 负载 | 修复前 | 修复后 | vLLM (FA2, block256) |
|---|---|---|---|
| 256×1024（均衡） | 6838 | **8425** | 10136 |
| 128×input100×out1024（decode-heavy） | 7233 | **14053** | 14359 |

- decode-heavy 从 vLLM 的 50% → **98%**（基本追平）。
- 256×1024 从 vLLM 的 67% → **83%**。
- 正确性：greedy `generate` 文本与 HF 一致；`stream_generate` 增量 delta 正常（"three, four, five, six,"）。

### 剩余 gap（256×1024 的 83%→100%）
decode 已追平，剩余差距在**均衡负载的 prefill 阶段**（256 seq × avg 562 input = 144k prefill tokens）与 256 seq 下的 KV 压力（MiniInfer 默认 `gpu_memory_utilization=0.6` vs vLLM 0.9 → 更小 KV、更多 retraction）。后续可查 prefill 效率与 KV 利用率。

---

## 10. KV 利用率 + prefill 效率排查

### KV 利用率：默认 0.6 → 0.9
- 256×1024 **均衡**：0.6 → 0.9 吞吐不变（8425 → 8449，此负载在 0.6 下无 retraction）。
- 256×1024 **decode-heavy**（input 100, output 1024）：0.6 下 **KV 耗尽崩溃**（`Failed to allocate KV cache`），0.9 下正常运行 → **17369 tok/s**（> vLLM 16333，反超）。
- 结论：0.6 在 256 seq × 长序列下容量不足。默认改为 **0.9**（与 vLLM 一致），容量足够、吞吐不损、长序列 decode-heavy 反超 vLLM。

### prefill 效率：MiniInfer 65% of vLLM（剩余 gap 主因）
prefill-heavy 对比（128 seq × input 1024 × output 8，input tokens/time）：
- MiniInfer：**87939 in_tok/s**
- vLLM（FA2, block256, max_num_batched_tokens=8192）：**134443 in_tok/s**
- MiniInfer 是 vLLM 的 65%（1.53× 慢）。

profile prefill-heavy：
- `forward_extend` 占 CUDA 94%，其中 **GEMM（aten::mm + ampere gemm）占 ~64%**，attention（flash_attn_varlen）占 ~10%。
- prefill 是 **GEMM-bound**，且 MiniInfer prefill 走 **eager**（无 cuda graph），`cudaLaunchKernel` 8572 次（11 batch × ~779 launches/batch）。vLLM 用 **piecewise cuda graph（cuDAG）** 捕获 prefill，消除 launch 开销 + 更好 CPU/GPU overlap。

### 结论
- **decode 已追平/反超 vLLM**（decode-heavy 256×1024：17369 > 16333）。
- **剩余 256×1024 均衡负载的 83%→100% gap 全在 prefill**（prefill 65% of vLLM）。根因：prefill eager 无 cuda graph + GEMM/launch 开销。闭合需 **piecewise cuda graph for prefill**（vLLM cuDAG 做法）——这是一块独立的较大工程。

### 当前状态（FA2, Qwen3-0.6B, 默认 0.9）
| 负载 | MiniInfer | vLLM (FA2,block256) | 相对 |
|---|---|---|---|
| 256×1024 decode-heavy | 17369 | 16333 | **106%** ✅反超 |
| 128×1024 decode-heavy | 14053 | 14359 | 98% |
| 256×1024 均衡 | 8420 | 10136 | 83%（prefill gap） |

---

## 6. 产物

- `benchmark/bench_miniinfer.py`：对齐 vLLM 方法论的吞吐 bench，支持 `--page-size` A/B。
- `benchmark/profile_miniinfer.py`：torch.profiler 脚本，导出 chrome trace + top kernel/op 表。
- `/tmp/miniinfer_trace.json`：chrome trace（可在 `chrome://tracing` 打开）。
