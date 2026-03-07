# CPU-GPU Overlap 深度分析

## 问题现象

从最新的性能 trace 可以看到：
- ✅ **CUDA 同步箭头已消除**（`non_blocking=True` 优化生效）
- ❌ **CPU scheduling 仍未完全隐藏在 GPU 计算中**
- ❌ `forward_batch_init` 和 `schedule` 阶段仍占用可见的 CPU 时间

## 为什么消除 CUDA Sync 还不够？

### 问题 1: 异步传输的"假象"

**当前实现**:
```python
# 源数据在普通 CPU memory 中
tensor_cpu = torch.as_tensor(data, dtype=torch.int64)  
tensor_gpu = tensor_cpu.to(device, non_blocking=True)
```

**PyTorch 内部实际发生的事情**:
1. 检查源 tensor 是否在 pinned memory
2. 如果不是，**同步地**将数据复制到临时 pinned buffer
3. 然后才异步传输到 GPU

**结果**: 虽然没有显式的 CUDA sync，但步骤 2 的"同步复制到 pinned buffer"仍然占用 CPU 时间，阻碍 overlap。

### 问题 2: Python 元数据构建本身耗时

即使所有 GPU 操作都是异步的，Python 代码本身需要执行时间：

```python
# prepare_for_extend 中的 Python 循环
extend_ids = [r.fill_ids[len(r.prefix_indices) :] for r in batch.reqs]  # ~50μs
seq_lens = [len(r.fill_ids) for r in batch.reqs]                        # ~20μs
prefix_lens = [len(r.prefix_indices) for r in batch.reqs]                # ~20μs
extend_lens = [r.extend_input_len for r in batch.reqs]                   # ~20μs

# 嵌套循环展平 token IDs
extend_ids_flat = [token_id for ids in extend_ids for token_id in ids]  # ~100μs (大 batch)

# 多次 tensor 创建
torch.as_tensor(seq_lens, ...)      # ~10μs
torch.as_tensor(prefix_lens, ...)   # ~10μs
# ... 重复多次
```

**总计**: 单个 `prepare_for_extend` 可能需要 200-500μs 的纯 CPU 时间。即使 GPU 在异步执行，CPU 的这些工作仍然串行完成，占用调度时间。

### 问题 3: 单线程执行限制

当前架构是**单线程串行**:

```
Step N:
  1. process_pending(N-1)          ← CPU 工作（采样、detokenize）
  2. schedule(N)                   ← CPU 工作（调度逻辑）
  3. prepare_metadata(N)           ← CPU 工作（prepare_for_extend/decode/mixed）
  4. forward_batch_init(N)         ← CPU 工作（构建 ForwardBatch）
  5. run_async(N)                  ← 启动 GPU
  6. [GPU 执行 forward(N)]
  7. 循环到 Step N+1

Timeline:
GPU:  |---------- forward(N-1) ---------|---------- forward(N) ---------|
CPU:  |-- idle --|-- step 1-5 for N --|-- idle --|-- step 1-5 for N+1 --|
                 ↑                     ↑
              CPU 工作无法与 GPU 并行
```

**问题**: 步骤 2-4（schedule + prepare + init）在主线程上串行执行，无法与 GPU 的 forward(N-1) 真正并行。

### 问题 4: Overlap Executor 只覆盖部分工作

当前的 `overlap_executor` 只在 GPU 执行时处理**上一个 batch 的结果**:
- ✅ 覆盖: `process_pending()` (sampling, detokenize)
- ❌ 未覆盖: `schedule()`, `prepare_metadata()`, `forward_batch_init()`

这意味着最关键的 CPU 开销（元数据构建）仍然在主线程上同步执行。

## 真正实现 Overlap 的三个层次

### Level 1: 消除 CUDA 同步点 ✅ (已完成)

- 使用 `non_blocking=True`
- 避免 `.item()` 调用
- 使用 `torch.as_tensor()` 而非 `torch.tensor(..., device='cuda')`

**效果**: 消除显式的 GPU ↔ CPU 同步，GPU 不会等待 CPU

### Level 2: 真正的异步传输 🔄 (推荐下一步)

- **预分配 pinned memory buffer pool**
- 直接在 pinned buffer 上操作数据
- 实现零拷贝异步传输

**效果**: 
- CPU → GPU 传输更快（无临时复制）
- 减少内存分配开销
- 预计 forward_batch_init 再减少 20-30%

**实现**: 见 `Pinned_Memory_优化方案.md`

### Level 3: 后台线程预先构建元数据 🔄 (终极优化)

在 GPU 执行 forward(N) 时，后台线程并行准备 batch(N+1) 的元数据:

```
Timeline (理想状态):
GPU:     |-------- forward(N) --------|-------- forward(N+1) --------|
Thread1: |-- prepare(N+1) --|-- idle --|-- prepare(N+2) --|
Thread2: |-- process(N) --|-- idle --|-- process(N+1) --|
```

**实现策略**:
1. 引入 `MetadataBuilder` 后台线程
2. 使用 `ThreadPoolExecutor` 或专用 worker 线程
3. 预分配**两套** pinned buffer (double buffering)
4. 在 `run_async(N)` 后立即提交 `prepare(N+1)` 任务

**难点**:
- Python GIL 限制（可考虑用 C++ extension 或 Numba 绕过）
- 线程安全（buffer 访问需要同步）
- 调度复杂度（需要预判下一个 batch 的内容）

## SGLang 的实现

SGLang 使用了以下技术：

1. **Pinned Memory Pool**: 预分配所有元数据 buffer
2. **StreamExecutor**: 在独立的 CUDA stream 上执行 forward
3. **Minimal Sync**: 只在必须的地方同步（process_pending 前）
4. **Cpp Extension**: 部分热点路径用 C++ 实现，绕过 GIL

但 SGLang 仍然是**单线程**的，未使用多线程预构建元数据。它的 overlap 主要来自：
- GPU forward(N) 与 CPU process(N-1) 并行
- 高效的 pinned memory 使用
- 优化的 Python 循环（减少到最少）

## 优化优先级建议

### 短期（1-2 天）
1. ✅ 消除 CUDA 同步点（已完成）
2. 🔄 **实现 pinned memory buffer pool** ← 推荐立即实施
   - 投入产出比高
   - 实现相对简单
   - 预计 20-30% 额外提升

### 中期（1-2 周）
3. 🔄 优化 Python 循环
   - 用 Numba JIT 或 C++ extension 重写热点函数
   - `prepare_for_extend/decode/mixed` 的 list comprehension
   - `ForwardBatch.init_new` 的采样参数提取

### 长期（1 个月+）
4. 🔄 后台线程预构建元数据
   - 引入 worker 线程 pool
   - 实现 double buffering
   - 需要重构调度器架构

## 当前瓶颈的定量估计

基于 trace 图估算（假设 batch size = 32, seq_len = 512）:

| 阶段               | 当前耗时 | Level 2 优化后 | Level 3 优化后  |
| ------------------ | -------- | -------------- | --------------- |
| schedule           | ~200μs   | ~200μs         | ~0μs (后台完成) |
| prepare_metadata   | ~300μs   | ~200μs         | ~0μs (后台完成) |
| forward_batch_init | ~150μs   | ~100μs         | ~0μs (后台完成) |
| GPU forward        | ~10ms    | ~10ms          | ~10ms           |
| **CPU 空闲比例**   | **94%**  | **95%**        | **99%+**        |

**结论**:
- Level 1 (当前): CPU 工作占 ~6% 的时间，GPU 利用率 94%
- Level 2 (pinned memory): CPU 工作降至 ~5%，GPU 利用率 95%
- Level 3 (后台线程): CPU 工作几乎为 0，GPU 利用率 99%+

对于 10ms 的 forward 时间，当前 600μs 的 CPU 开销已经相对较小。**除非 GPU 计算非常快（<2ms），否则 Level 2 优化即可满足需求。**

## 总结

**为什么消除 CUDA sync 后仍然不够**:
1. 非 pinned memory 的异步传输仍有隐藏开销
2. Python 元数据构建本身需要时间（200-500μs）
3. 单线程架构限制了真正的并行

**下一步行动**:
1. **立即实施**: Pinned memory buffer pool（投入产出比最高）
2. **视情况实施**: Python 循环优化（如果元数据构建仍是瓶颈）
3. **慎重考虑**: 后台线程（架构改动大，收益有限）

**关键认知**:
- 对于中等规模的模型（forward ~10ms），当前的优化（Level 1）已经接近理论极限
- 继续优化的收益递减：从 94% → 95% → 99% GPU 利用率
- **应该将优化重点转向其他瓶颈**（如模型并行、量化、kernel 优化等）
