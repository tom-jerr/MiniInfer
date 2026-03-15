# CPU Schedule 优化总结

## 问题

从性能分析图可以看到，CPU 的 schedule 操作没有完全隐藏在 GPU 计算中，导致 GPU 出现空闲等待。

## 根因分析

参考 SGLang 的设计，发现主要瓶颈在 CPU 侧元数据构建使用了大量 Python 循环，无法与 GPU 计算有效 overlap：

1. **prepare_for_mixed/decode**: 使用 list comprehension 构建元数据
2. **\_rebuild_running_batch_metadata**: 循环计算每个请求的 seq_len
3. **ForwardBatch.init_new**: 多次 list → tensor 转换

## 优化方案

### 核心思想

- **缓存替代计算**: 在 Req 对象中维护缓存字段，避免重复计算
- **批量替代循环**: 使用预分配的 tensor 和向量化操作替代 Python 循环
- **减少转换**: 直接在 GPU 上分配 tensor，减少 CPU ↔ GPU 数据转移

### 具体实现

#### 1. Req 类增强 (scheduler_batch.py)

```python
# 新增缓存字段
self._cached_seq_len = len(token_ids)     # 缓存序列长度
self._cached_last_token = None            # 缓存最后的 token

# 新增方法
def append_output_token(self, token_id: int):
    """追加 token 并自动更新缓存"""
    self.output_ids.append(token_id)
    self._cached_seq_len += 1
    self._cached_last_token = token_id

def get_last_token(self) -> Optional[int]:
    """获取最后一个 token（使用缓存）"""
    return self._cached_last_token or (self.output_ids[-1] if self.output_ids else None)

@property
def current_seq_len(self) -> int:
    """获取当前序列长度（使用缓存）"""
    return self._cached_seq_len
```

#### 2. 优化 \_rebuild_running_batch_metadata (scheduler.py)

```python
# 优化前：Python 循环 + 重复计算
seq_lens = [len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs]
batch.seq_lens = torch.tensor(seq_lens, ...)

# 优化后：预分配 + 缓存访问
seq_lens = torch.empty(bs, dtype=torch.int64, device=device)
for i, req in enumerate(batch.reqs):
    seq_lens[i] = req.current_seq_len  # 使用缓存
```

#### 3. 优化 prepare_for_decode (kv_cache_manager.py)

```python
# 优化前：list comprehension + 双重转换
last_tokens = [req.output_ids[-1] for req in batch.reqs]
batch.input_ids = torch.tensor(last_tokens, ...).to(device)

# 优化后：直接在 GPU 上分配 + 缓存访问
last_tokens = torch.empty(bs, dtype=torch.int64, device=self.device)
for i, req in enumerate(batch.reqs):
    last_tokens[i] = req.get_last_token()  # 使用缓存
batch.input_ids = last_tokens
```

#### 4. 优化 prepare_for_mixed (kv_cache_manager.py)

```python
# 优化前：多次遍历
extend_num_tokens = sum(len(ids) for ids in extend_ids_list)
for r in decode_reqs:
    cur_seq_len = len(r.origin_input_ids) + len(r.output_ids)

# 优化后：边循环边累加 + 使用缓存
extend_num_tokens = 0
for r in extend_reqs:
    extend_num_tokens += len(ids)  # 边循环边累加

for r in decode_reqs:
    cur_seq_len = r.current_seq_len  # 使用缓存
```

#### 5. 删除废弃代码 (scheduler_batch.py)

删除了包含 Python 循环的旧实现 `compute_position_torch`，保留向量化的 `compute_positions_extend`。

## 改动的关键文件

1. ✅ `miniinfer/scheduler/scheduler_batch.py` - Req 类增强
2. ✅ `miniinfer/scheduler/scheduler.py` - 优化 \_rebuild_running_batch_metadata
3. ✅ `miniinfer/kvcache/kv_cache_manager.py` - 优化 prepare_for_mixed/decode
4. ✅ `miniinfer/engine/llm_engine.py` - 更新 token append 调用

## 预期效果

### 定量预测

- **forward_batch_init 耗时**: 减少 50-70%
- **CPU 操作次数**: 减少 40-60%
- **内存分配次数**: 减少 30-50%

### 定性改进

- ✅ 更好的 CPU-GPU overlap
- ✅ 减少 GPU 空闲等待时间
- ✅ 提升整体 throughput
- ✅ 代码更简洁易维护

## 验证方法

### 功能测试

```bash
# 运行测试套件确保无回归
pytest tests/test_stream_generate.py
pytest tests/test_cuda_graph_decode.py
pytest tests/test_memory_pressure.py
```

### 性能测试

```bash
# 对比优化前后的性能
python benchmark/bench_simple.py --model Qwen/Qwen2-1.5B-Instruct
```

重点关注：

- forward_batch_init 阶段耗时
- schedule 阶段耗时
- GPU 利用率
- Throughput (tokens/s)

## 下一步优化方向

1. 🔄 **预分配 buffer pool**: 为 sampling params 等创建全局 buffer
2. 🔄 **使用 Numba/C++ 扩展**: 加速更多热点 CPU 操作
3. 🔄 **优化 radix cache 操作**: 减少前缀树查找开销
4. 🔄 **异步元数据构建**: 使用多线程预构建下一批次的元数据

## 参考

- SGLang scheduler 设计: https://github.com/sgl-project/sglang
- vLLM continuous batching: https://github.com/vllm-project/vllm
- 本项目的 overlap executor 实现: `miniinfer/engine/overlap_executor.py`
