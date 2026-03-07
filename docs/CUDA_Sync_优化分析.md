# CUDA 同步点分析和优化方案

## 问题

从性能分析图可以看到 `forward_batch_init` 和 `run_async` 阶段有大量 CUDA 同步（箭头），阻止了 CPU-GPU 完全 overlap。

## 主要 CUDA 同步来源

### 1. `torch.tensor(..., device='cuda')` 
**位置**: kv_cache_manager.py 多处
```python
torch.tensor(extend_prefix_lens, dtype=torch.int64).to(self.device)
torch.tensor(all_ids, dtype=torch.int64, device=self.device)
```
**问题**: 直接创建 GPU tensor 会触发同步的 CPU→GPU 传输

### 2. `.to(device)` 不带 `non_blocking=True`
**位置**: kv_cache_manager.py
```python
torch.tensor(...).to(self.device)  # 同步传输！
```

### 3. `.item()` 调用
**位置**: scheduler_batch.py:282
```python
max_seq_len = int(batch.seq_lens_cpu.max().item())  # 不必要的 .item()
```
**问题**: 虽然是 CPU tensor，但 .max().item() 仍可能触发同步

### 4. Python list → GPU tensor 转换
**位置**: scheduler_batch.py:307-320, kv_cache_manager.py 多处
```python
torch.tensor([r.sampling_params.temperature for r in batch.reqs], device=dev, ...)
```
**问题**: List comprehension + 直接创建 GPU tensor

## SGLang 的优化方案

### 核心原则
1. **使用 pinned memory**: 预分配 pinned CPU buffer，异步传输到 GPU
2. **non_blocking=True**: 所有 CPU→GPU 传输都使用异步模式
3. **避免 GPU→CPU 同步**: 在 CPU tensor 上完成所有计算
4. **延迟传输**: 尽可能晚地将数据传输到 GPU

### 具体实现

#### 1. 预分配 pinned memory buffer
```python
class KVCacheManager:
    def __init__(self, ...):
        # 预分配 pinned memory buffers
        self.pinned_prefix_lens = torch.empty(max_num_seqs, dtype=torch.int64, pin_memory=True)
        self.pinned_seq_lens = torch.empty(max_num_seqs, dtype=torch.int64, pin_memory=True)
        self.pinned_input_ids = torch.empty(max_extend_tokens, dtype=torch.int64, pin_memory=True)
```

#### 2. 使用 torch.as_tensor 和 non_blocking
```python
# 优化前（同步）
tensor = torch.tensor(python_list, dtype=torch.int64, device='cuda')

# 优化后（异步）
tensor_cpu = torch.as_tensor(python_list, dtype=torch.int64)
tensor_gpu = tensor_cpu.to('cuda', non_blocking=True)

# 或使用预分配的 pinned buffer
self.pinned_buffer[:len(python_list)].copy_(torch.as_tensor(python_list))
tensor_gpu = self.pinned_buffer[:len(python_list)].to('cuda', non_blocking=True)
```

#### 3. CPU tensor 上完成计算
```python
# 优化前（可能同步）
max_seq_len = int(batch.seq_lens_cpu.max().item())

# 优化后（纯 CPU 计算）
max_seq_len = int(batch.seq_lens_cpu.max())  # 直接从 CPU tensor 取值
```

#### 4. 分离 tensor 创建和传输
```python
# 优化前（一次性创建 GPU tensor）
extend_ids_tensor = torch.tensor(all_ids_extend, dtype=torch.int64, device=self.device)

# 优化后（先 CPU，再异步传输）
extend_ids_cpu = torch.as_tensor(all_ids_extend, dtype=torch.int64)
extend_ids_tensor = extend_ids_cpu.to(self.device, non_blocking=True)
```

## 实施计划

### Phase 1: 消除直接的 GPU tensor 创建
- ✅ kv_cache_manager.py: 所有 `torch.tensor(..., device=device)` 改为两步
- ✅ scheduler_batch.py: ForwardBatch.init_new 中的 tensor 创建

### Phase 2: 添加 non_blocking=True
- ✅ 所有 `.to(device)` 调用添加 `non_blocking=True`

### Phase 3: 消除不必要的 .item()
- ✅ 将 `max_seq_len = int(batch.seq_lens_cpu.max().item())` 改为直接访问

### Phase 4: 预分配 pinned memory buffers（可选，性能提升显著）
- ⏳ 在 KVCacheManager 中添加 pinned buffer pool
- ⏳ 在 Scheduler 中复用 buffers

## 参考

- SGLang scheduler: [sglang/python/sglang/srt/managers/scheduler.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) 
- SGLang TP worker: [sglang/python/sglang/srt/managers/tp_worker.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tp_worker.py)
- 关键点：`prepare_input_metadata` 函数使用 pinned buffer + non_blocking
