# Overlap + CUDA-Graph 竞态修复与排查

对应提交：`8c064dc fix: overlap + CUDA-graph stream race causing intermittent decode token flips`

本文档记录 overlap 调度与 CUDA graph replay 之间的流竞态——从现象、二分定位、根因、到修复方案选型（含吞吐对比）与验证的完整过程。

---

## 1. 现象

修复 greedy 采样乱码（见 `Logger_Loader_MoE_与采样器乱码修复.md` §5）后，默认配置（`enforce_eager=False, enable_overlap=True`）下 greedy 解码仍**偶发** token 翻转：

```
HF   : [358, 2776, 14589, 369, 279, 60009, 13, 358, 2776, 14589, 369, 279, 60009, 13, 358, 2776]
run1 : [358, 2776, 14589, 1,    279, 60009, 1,  358, 2776, 1,    1,  1,  1,     13, 358, 2776]
run2 : [358, 1,    14589, 369, 279, 60009, 13, 358, 2776, 14589,369, 279,1,     1,  358, 2776]
run3 : [358, 2776, 14589, 369, 279, 60009, 13, 358, 2776, 14589,369, 279,60009, 13, 358, 2776]  ← 偶尔全对
```

特征：
- **间歇性**：同样输入，3 次运行结果不同（run1 差 6 个、run2 差 3 个、run3 全对）→ 典型**竞态**。
- 翻转 token 几乎总是 `1`（`"`），提示读到 stale/未初始化 buffer。
- 翻转后后续 token 又能「追平」HF（run1 step 2 错成 1，但 step 3-7 与 HF 一致）→ 状态本身没坏，是某次读取拿到旧值。

## 2. 二分定位

### 2.1 四象限组合测试
固定 prompt = `"Hello, how are you?"`，对比 HF：

| 配置 | 结果 |
|---|---|
| eager + no-overlap | ✅ 16/16 bit-exact |
| eager + overlap | ✅ 16/16 bit-exact（3/3） |
| graph + no-overlap | ✅ 16/16 bit-exact |
| **graph + overlap（默认）** | ❌ 间歇翻转 |

→ 竞态**只**在 graph + overlap 组合下出现。eager 全对、graph 单独全对、overlap 单独全对。

> 关键推论：overlap 的 future_map / placeholder 机制没问题（eager+overlap 全对）；KV cache 读取没问题（graph+no-overlap 全对）；问题出在 **CUDA graph replay 在 overlap 多流环境下的行为**。

### 2.2 instrumentation 的陷阱
初版排查给 `cuda_graph_runner.replay` 加了 `.cpu().tolist()` 打印每步 input_ids / logits argmax：

```
replay step 0: input_ids=[358]    logits_argmax=[2776]
replay step 1: input_ids=[2776]   logits_argmax=[14589]
replay step 2: input_ids=[14589]  logits_argmax=[369]   ← 全对！
```

两个模式都「全对」——但去掉 instrumentation 后竞态依旧。

→ **`.cpu().tolist()` 的 GPU 同步掩盖了竞态**。这反过来证明：竞态是「graph replay 的产出还没就绪就被消费」类型的同步缺失。

### 2.3 同步二分
在 `replay` 的 `captured.graph.replay()` 之后插不同同步，跑 3 次：

| 同步方式 | 结果 |
|---|---|
| 无 | ❌ 间歇翻转 |
| `torch.cuda.synchronize()`（全设备） | ✅ 3/3 |
| `torch.cuda.current_stream().synchronize()`（仅 forward_stream） | ✅ 3/3 |

→ 仅同步 forward_stream 就能修复。`forward_stream.synchronize()` 会**阻塞 CPU 线程**，从而阻止 step N+1 的 schedule_stream 提交——即把 schedule 与 forward 串行化了。这暗示竞态是 **schedule_stream（step N+1 准备）与 forward_stream（step N graph replay）之间的跨流干扰**。

## 3. 根因

overlap 三流模型（SGLang 风格）：

```
schedule_stream: |-- prepare(N) --|-- prepare(N+1) --|   # 调度 + H2D + 元数据
forward_stream :       |-- forward(N) --|-- forward(N+1)--|   # graph replay + sample
copy_stream    :               |-- copy(N) --|-- copy(N+1)--|   # D2H
```

overlap 的收益正在于 `prepare(N+1)`（schedule_stream）与 `forward(N)`（forward_stream）**并行**。

在 graph 模式下，`forward(N)` 是一次 `captured.graph.replay()`。replay 读取：
- `captured.*` 静态缓冲（forward_stream 上由 `copy_` 填充，同流有序）；
- KV cache（key_cache/value_cache，step N 的行，只读）；
- 模型权重（只读）。

而 `schedule_stream` 的 `prepare(N+1)`（`ForwardBatch.init_new`）会做 H2D 拷贝、`compute_positions_extend` 等 GPU 计算、以及 caching allocator 分配——这些与 forward_stream 的 graph replay 在**不同流上并行**，干扰了 graph replay 的数据访问，偶发读到 stale 值 → 错误 logits → token 1。

> 注：未能精确指认是哪一块共享状态被踩（静态审查显示 graph 只读持久 `captured.*` + KV，而 `init_new` 只创建新 tensor）。怀疑是 PyTorch caching allocator / CUDA graph 在并发流下的内存序细节。但二分已明确：**只要 schedule_stream 的 GPU 活动不和 graph replay 并行，竞态即消失**。因此修复方向是给这两段加正确的流屏障，而不是去追底层内存序。

eager + overlap 不受影响：eager forward 是普通 kernel 序列，不触发该 graph 相关的敏感内存路径。

## 4. 修复方案选型

候选方案及吞吐对比（Qwen3-0.6B，4 seq × 128 token，greedy）：

| 方案 | tok/s | 正确 | 说明 |
|---|---|---|---|
| racy graph+overlap（修复前） | 838 | ❌ | 原默认 |
| **graph+overlap+`wait_stream`（采用）** | **774** | ✅ | 非阻塞流屏障 |
| graph + no-overlap | 708 | ✅ | 直接关 overlap |
| graph+overlap+`forward_stream.synchronize()` | 564 | ✅ | 阻塞 CPU，overlap 失效反而更慢 |
| eager + overlap | 131 | ✅ | 丢 graph，4× 慢 |

### 采用方案：`schedule_stream.wait_stream(forward_stream)`

`llm_engine.py: step_overlap` Phase 3 之前：

```python
schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
# When CUDA graph is in use, schedule_stream's GPU prep (H2D copies, position
# compute, allocations) for step N+1 must not race with forward_stream's graph
# replay for step N. Make schedule_stream wait for the last forward_stream op
# (the graph replay) before issuing its own GPU work. This is non-blocking on
# the CPU (unlike forward_stream.synchronize()), preserving CPU post-processing
# overlap with GPU forward.
if schedule_stream is not None and self.model_runner.use_cuda_graph:
    schedule_stream.wait_stream(self.overlap_executor.forward_stream)
```

要点：
- **仅 graph 模式启用**（`use_cuda_graph` 守卫）：eager + overlap 不受影响，保留完整 overlap。
- **非阻塞**：`wait_stream` 只往 schedule_stream 提交一个 wait op，不阻塞 CPU 线程。CPU 后处理（detokenize、输出处理）仍与 GPU forward 重叠。
- 代价：`prepare(N+1)` 的 GPU 部分（H2D、position 计算）不再与 `forward(N)` 的 GPU 重叠——这部分被串行化。但 CPU 后处理重叠仍在，所以比「关 overlap」（708）还快（774）。
- 比 `forward_stream.synchronize()`（564）好得多：后者阻塞 CPU，连 CPU 后处理重叠也丢了，反而比关 overlap 更慢。

> 为什么 `wait_stream` 比 `synchronize` 快？`synchronize` 让 CPU 线程空转等 GPU，期间既不能做后处理也不能提交 schedule；`wait_stream` 只让 schedule_stream 的 GPU work 等 forward_stream，CPU 线程继续做后处理与 schedule 的 CPU 部分。

## 5. 验证

### 5.1 正确性
默认配置（`enforce_eager=False, enable_overlap=True`）下，3 个 prompt × 24 token 与 HF **逐 token bit-exact**：

```
prompt 0: match=True  mini: [358, 2776, 14589, 369, 279, 60009, 13, 358, 2776, 14589, 369, 279]
prompt 1: match=True  mini: [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315]
prompt 2: match=True  mini: [11, 1052, 1033, 220, 18, 15, 15, 15, 1251, 304, 264, 6290]
ALL MATCH: True
```

且单 prompt × 16 token 连跑 3 次均全对（修复前 3 次里 2 次错）。

### 5.2 吞吐
774 tok/s，约为 racy 版本（838）的 92%，且优于所有其它正确方案。

## 6. 遗留 / 后续

- 未精确指认底层共享状态（疑为 caching allocator / CUDA graph 并发流内存序）。当前 `wait_stream` 屏障正确且性能可接受。若后续要榨回最后 ~8% 吞吐，可深挖具体踩踏点（例如把 graph replay 的输入 snapshot 进一步隔离，或用独立 graph memory pool）。
- 该屏障只覆盖 graph 模式；若未来 eager 路径也出现类似跨流问题，可类比处理。
