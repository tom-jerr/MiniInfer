# Logger / 多线程权重加载 / Qwen3-MoE / 崩溃日志 / 采样器乱码修复

对应提交：`c684ced feat: vLLM-style logger, multi-threaded weight loader, Qwen3-MoE, crash logging; fix greedy sampler`

本文档记录本轮第一笔提交的五块改动：vLLM 风格 logger、多线程权重加载、Qwen3-MoE 支持、推理崩溃上下文日志、以及 greedy 采样乱码修复（含逐算子排查过程）。

---

## 1. vLLM 风格 logger

### 背景
原代码用原生 `logging.getLogger(__name__)`，且包内散落 58 处 `print()` 调试输出，格式不统一、不可 grep，多进程（overlap worker）下无法区分来源。

### 原理
移植 nano-vllm 的 `nanovllm/utils/logger.py`，输出 vLLM V1 风格：

```
(EngineCore pid=12345) INFO  07-01 13:34:17 [model_runner.py:95] Model loaded on device cuda.
```

- `(Role pid=PID)` 前缀：多进程下区分来源，默认 role `EngineCore`，可由 `set_process_role` 覆盖。
- 颜色：按 no-color.org 规范，`NO_COLOR` 永远胜出；其次 `MINIINFER_LOG_COLOR`、`CLICOLOR_FORCE`、TTY 检测。非 TTY（管道/文件）自动去色。
- 单一 `StreamHandler`，`propagate=False`，子模块共享格式。

### 修改
- 新增 `miniinfer/utils/logger.py`：`_MiniInferFormatter`、`get_logger`、`set_process_role`，环境变量由 `NANOVLLM_LOG_COLOR` 改名 `MINIINFER_LOG_COLOR`。
- `miniinfer/utils/__init__.py` 导出 `get_logger`、`set_process_role`。
- 包内所有 `import logging; logger = logging.getLogger(__name__)` → `from miniinfer.utils import get_logger; logger = get_logger(__name__)`。
- 包内所有 `print(...)` 按语义分级：形状/调试 dump → `logger.debug`；进度/状态 → `logger.info`；错误 → `logger.error`。`memory_utils`（用 `logging.Logger` 做类型注解）与 `scheduler`（用 `logging.DEBUG`）保留 `import logging`。

### 验证
```
(EngineCore pid=64066) INFO  07-01 13:34:17 [<string>:4] Starting to load model /root/models/Qwen3-0.6B
(EngineCore pid=64066) WARNING 07-01 13:34:17 [<string>:5] low GPU memory
```
`MINIINFER_LOG_COLOR=1` 出彩色；管道重定向自动去色。

---

## 2. 多线程权重加载

### 背景
`miniinfer/loader/weight.py` 的 `load_hf_weight` 单线程顺序逐 shard 读 safetensors，无 page-cache 预热，大模型加载慢。

### 原理（移植 nano-vllm `utils/loader.py`）
两阶段重叠：

1. **Prefetch（后台线程池）**：8 个后台线程把每个 shard 顺序按 16MiB 大块读一遍，不保留字节，只预热 OS page cache，使后续 `safe_open` 的随机 mmap 访问命中内存而非磁盘。顺序大块读远快于 `safe_open` 的惰性 mmap，NFS/Lustre 上尤其明显。
2. **逐 tensor 流式读取**：主线程对每个 shard `safe_open` 后 `get_tensor(name)` 一次取一个 tensor，峰值内存低。Prefetch 在后台与该迭代重叠。

> 为什么不直接多线程整文件 `load_file`？瓶颈是磁盘带宽，几个并发读者就饱和了；再多的线程只会增加争用与内存压力。Prefetch + 逐 tensor 读攻克的正是真正瓶颈（冷 page cache + 随机 mmap）。

保留 MiniInfer 的「合并 state_dict」对外接口（dense 模型 `load_weights(state_dict)` 签名不变）。

### 修改
- `_shard_files` / `_prefetch_file`（16MiB 块顺序读）/ `_start_prefetch`（daemon 线程 + `ThreadPoolExecutor`）/ `_iter_weights`（流式 yield `(name, tensor)`，与 prefetch 重叠）。
- `load_hf_weight` 改为流式填 state_dict + 后台 prefetch；加 vLLM 风格日志（shard 数、加载耗时）。
- `_merge_state_dict` **新增跳过 `.experts.` 键**：MoE per-expert 权重（`mlp.experts.{e}.{gate|up|down}_proj.weight`）不参与 qkv/gate_up 合并，保持原样由 MoE 模块自行堆叠（见 §3）。
- `loader/weight.py` 取 logger 改为延迟 `_log()`，避免 loader（被 layers/models 早期导入）与 `miniinfer.utils` 之间的静态循环导入。

### 验证
Qwen3-0.6B 加载 `Model loaded in 0.57s (227 tensors)`；合成 tiny 模型 0.04s。merge 单测：`.experts.` 键原样保留，`mlp.gate.weight` 保留，attention q/k/v 合并为 `qkv_proj`。

---

## 3. Qwen3-MoE 支持（移植 nano-vllm，保留 TP/EP，去量化）

### MoE 前向原理
```
hidden (T, H)
  ├─ gate = Linear(H → num_experts)          # router, replicated
  ├─ probs = softmax(gate)
  ├─ topk_weights, topk_ids = topk(probs, K) # (T, K)
  ├─ if norm_topk_prob: topk_weights /= sum
  ├─ FusedExperts(hidden, topk_ids, topk_weights)  # 专家计算
  └─ (TP/EP all-reduce 合并)
```

### 权重堆叠布局（核心）
每专家有 `gate_proj / up_proj / down_proj`，堆叠成两个 3D 张量以用一次 grouped GEMM 算完：
```
w13_weight: (E_local, 2*I_local, H)   # gate||up 沿 dim=0 拼接，行被 TP 切
w2_weight : (E_local, H, I_local)     # down，列被 TP 切
```
`E_local = num_experts / ep_size`，`I_local = moe_intermediate_size / tp_size`。

### TP/EP 语义（保留）
- **EP**：每 rank 拥有 `[ep_rank*E/EP, (ep_rank+1)*E/EP)`；`expert_map[E]` 中本地专家填 local index、远程填 -1；`FusedExperts` 只算本地专家，远程 tile 输出 0，EP all-reduce 求和。
- **TP**：每专家 intermediate 维被 TP 切（w13 行切、w2 列切），输出 partial，TP all-reduce。
- `MoEDispatcher.combine` 在 `ep_size>1` 或 `tp_size>1` 时 `dist.all_reduce(SUM)`；size==1 时 no-op。MiniInfer 单 GPU 走 no-op，但结构与 nano-vllm 完全一致，未来接 dist 即可用。

### grouped GEMM（Triton）
`fused_experts_triton` 两段 GEMM + 一段 elementwise：
1. `moe_align_block_size`：把 `(T,K)` 个 (token,k) 对按专家稳定排序、每专家桶补齐到 `block_m` 倍数 → `sorted_token_ids / expert_ids / num_tokens_post_padded`。小 batch（numel≤1024 且 E≤64）走单 kernel `_small_batch_align_kernel`，大 batch 走 count→meta→fill→scatter 四 kernel，全程无 `.item()` host sync。
2. GEMM1：`A=hidden @ B=w13 → c1(T*K, 2I)`，`mul_routed_weight=False`。
3. SiLU·mul：`silu(c1[:,:I]) * c1[:,I:] → c2`（eager，廉价）。
4. GEMM2：`A=c2 @ B=w2 → c3(T*K, H)`，`mul_routed_weight=True`（把 topk 权重折进最终 reduce）。
5. reduce：`c3.view(T,K,H).sum(dim=1) → (T,H)`。

kernel 通过 `expert_ids[pid_m]` 取当前 tile 专家、`sorted_token_ids` 取 token 行；`expert_map[expert_ids]` 把全局专家 id 映射到本地 w13/w2 行；越过 `num_tokens_post_padded` 的 tile 直接 return。

### 修改（新增文件）
- `config/model/qwen3_moe.py`：`Qwen3MoeConfig`，解析 `num_experts`/`num_experts_per_tok`/`moe_intermediate_size`/`decoder_sparse_step`/`mlp_only_layers`/`first_k_dense_replace`（HF 可为 None，统一 `or 0`）/`norm_topk_prob`。
- `layers/fused_moe/dispatcher.py`：`MoEParallelConfig`、`MoEDispatcher`、`prepare_mlp`。
- `layers/fused_moe/fused_moe_kernel.py`：align + grouped GEMM（仅 BF16/FP16/FP32，去掉量化分支）。
- `layers/fused_moe/experts.py`：`FusedExperts`（堆叠 w13/w2、expert_map buffer、workspace 复用、triton + CPU eager fallback）。
- `layers/fused_moe/layer.py`：`FusedMoE`（router + experts + dispatcher，`load_expert_weight` 保留 TP narrow）。
- `models/fused_qwen3_moe.py`：`Qwen3MoeForCausalLM`，复用 `fused_qwen3.Qwen3Attention`；按 `first_k_dense_replace`/`decoder_sparse_step`/`mlp_only_layers` 选 MoE 或 dense MLP；MoE 块 `load_weights` 取 `mlp.gate.weight` + 逐专家 `load_expert_weight`。

### 关键细节：`mlp.gate` 别名
`Qwen3MoeSparseMoeBlock` 把 `self.gate = self.experts.gate`，使 HF 键 `mlp.gate.weight` 能解析到 router。否则 loader 找不到落点，router 停在随机初始化，topk 选随机专家，decode 出垃圾。

### 验证
- Qwen3-30B-A3B 真实 config 下 MoE block 构建/权重加载/Triton forward：`w13 (128,1536,2048)`、`w2 (128,2048,768)`，router topk 分散 126/128 专家。
- Triton vs eager 对齐：max diff ~6e-5（fp16）。
- 合成 tiny MoE 模型（2 层 4 专家）通过 `LLMEngine.generate` 跑通完整链路。

---

## 4. 推理崩溃上下文日志

### 背景
单凭 traceback 不足以定位推理崩溃——需要同时看调度器状态、KV cache 占用、当前 batch 元数据、attention backend 的 forward_metadata。

### 修改
新增 `miniinfer/utils/crash_logger.py`，`log_inference_crash(exc, scheduler, model_runner, forward_batch, scheduled_batch, stage)` 防御性 dump（全部 `getattr`，dump 自身不再抛异常）：
- **ModelRunner**：model 类型、model_type、config 各字段、MoE 字段、cuda graph 状态。
- **Scheduler**：waiting/running/pending 队列长度、逐 running req 状态、KV cache 占用（page_size、free_pages、available_size）。
- **ScheduledBatch / ForwardBatch**：forward_mode、batch_size、input_ids/positions/seq_lens/extend_lens/采样张量等（shape + 数值统计）。
- **Attention backend**：type、page_size、`forward_metadata` 的 block_table/cache_seqlens/max_seq_len_k 等。

tensor 只取 `shape/dtype/device + min/max/mean`，避免把整张 block table 打成日志。

在 `step` / `step_overlap` 的 schedule、forward_init、run_forward_async、run_sample_async、各 `process_pending_batch` / flush 循环处包 `try/except` → `log_inference_crash` → `raise`。不吞异常。

### 验证
真实异常路径触发，输出（节选）：
```
========== Inference step crashed [step_overlap.schedule] ==========
Exception: RuntimeError: Failed to allocate request pool slots: requested 1 slots, but request pool is full.
==== ModelRunner ====  model=Qwen3ForCausalLM model_type=qwen2 device=cuda use_cuda_graph=True
==== Scheduler state ====
  waiting_queue=0 running_batch.reqs=0 pending_release=2
  KVCache: device=cuda page_size=256 num_pages=? free_pages=1277 allocator.available_size=326912
========== End of crash context ==========
```
随后才是 Python traceback。

---

## 5. greedy 采样乱码修复（含逐算子排查）

### 现象
Qwen3-0.6B greedy（`temperature=0`）解码输出 `!!!!`（token 0/1 重复），与 transformers 参考实现完全不符。

### 排查过程：逐算子对比 HF
用 `output_hidden_states=True` 抓 HF 的 embedding + 每层输出，monkeypatch MiniInfer 抓 prefill 同位点，逐层比 max_abs：

| 算子 | MiniInfer vs HF |
|---|---|
| embedding | 完全一致（0.0） |
| 各层输入 L1→L27 | fp16 噪声，max_abs 0.002→0.25，相对误差 <1% |
| 末层 logits（last token） | **top-5 完全一致**，argmax=358 |

**结论：forward 完全正确**，乱码不在模型前向。而引擎输出 token 0，说明采样器没取 argmax。

### 复现采样器 bug
```python
probs = flashinfer.sampling.softmax(logits, temperature=[0.0])
# probs argmax=0, probs max=0.001  ← 均匀分布，不是 one-hot！
flashinfer.top_k_top_p_sampling_from_probs(probs, top_k=[0], top_p=[1.0])
# → [0]  ← 永远返回 0
```
两个 flashinfer 行为叠加：
1. `flashinfer.sampling.softmax(logits, temperature=0)` 返回**均匀分布**而非 one-hot（不像 PyTorch 的 `logits/0 → inf → softmax → one-hot`）。
2. `top_k_top_p_sampling_from_probs(probs, top_k=0)` 在 `top_k=0`（表示「禁用 top-k」）时**丢弃所有 token 返回 0**。

### 修复（`miniinfer/layers/sample.py`）
1. 概率改用 PyTorch 算：`softmax(logits.float() / temperature.clamp_min(1e-5))`，temp=0 → inf → one-hot，正确表达 greedy。
2. 无 top-k/top-p 过滤时走 **Gumbel-max**（`probs / Exp(1)` 的 argmax）：greedy（one-hot）返回 argmax、temp>0 是精确分类采样。全程向量化、无 GPU sync、无全词表排序。
3. 需要过滤时才用 flashinfer，但把 `top_k<=0` clamp 到 `vocab_size`（= 不过滤），绕开 flashinfer 的 top_k=0 bug。
4. 新增 `SamplingBatchInfo.enable_top_k_top_p`（CPU 端在 `ForwardBatch.init_new` 算好 `any(k>0) or any(p<1.0)`，避免 GPU `.any()` sync）做路径选择。

### 验证
- greedy (temp=0) eager：与 HF **逐 token 完全一致**（16/16）。
- temp=0.7 + top_p=0.9 + top_k=50：输出通顺。

---

## 附：attention backend 别名修复
`EngineConfig.attention_backend` 默认 `"flash_attention_2"`，而 `ModelRunner.init_attn_backend` 只认 `"flash_attn"`/`"flashinfer"`，导致示例开箱即崩。改为别名归一化：`flash_attn`/`flash_attention_2`/`flash_attention2`/`fa2` → FlashAttention2；`flashinfer`/`flash_infer` → FlashInfer。

## 已知遗留（下一笔提交修复）
overlap + CUDA-graph 组合偶发 decode token 翻转——见 `Overlap_CudaGraph_竞态修复.md`。
