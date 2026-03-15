# Pinned Memory Buffer Pool 优化方案

## 问题

虽然已消除 CUDA 同步点，但 CPU scheduling 仍未完全隐藏在 GPU 计算中。
根本原因：使用 `non_blocking=True` 但源数据不在 pinned memory，PyTorch 需要内部先复制到临时 pinned buffer。

## 优化目标

预分配 pinned memory buffer pool，实现真正的零拷贝异步传输，进一步减少 CPU 元数据构建开销。

## 实现方案

### 1. 在 KVCacheManager 中预分配 Pinned Memory Buffers

```python
class KVCacheManager:
    def __init__(self, ...):
        # 预分配 pinned memory buffers for metadata
        max_bs = self.max_num_seqs
        max_tokens_per_step = self.max_total_tokens // 4  # 估计值

        # 序列长度相关 (每个 batch 最多 max_bs 个请求)
        self.pinned_seq_lens = torch.empty(max_bs, dtype=torch.int64, pin_memory=True)
        self.pinned_prefix_lens = torch.empty(max_bs, dtype=torch.int64, pin_memory=True)
        self.pinned_extend_lens = torch.empty(max_bs, dtype=torch.int64, pin_memory=True)
        self.pinned_req_pool_indices = torch.empty(max_bs, dtype=torch.int64, pin_memory=True)

        # token IDs (extend 阶段可能有很多 token)
        self.pinned_input_ids = torch.empty(max_tokens_per_step, dtype=torch.int64, pin_memory=True)

        # Sampling params (per request)
        self.pinned_temperatures = torch.empty(max_bs, dtype=torch.float32, pin_memory=True)
        self.pinned_top_ps = torch.empty(max_bs, dtype=torch.float32, pin_memory=True)
        self.pinned_top_ks = torch.empty(max_bs, dtype=torch.int64, pin_memory=True)
```

### 2. 修改 prepare_for_extend 使用 pinned buffer

```python
def prepare_for_extend(self, batch: "ScheduledBatch"):
    # 1. 构建 Python list (CPU 侧，无 GPU 操作)
    extend_ids = [r.fill_ids[len(r.prefix_indices) :] for r in batch.reqs]
    seq_lens = [len(r.fill_ids) for r in batch.reqs]
    prefix_lens = [len(r.prefix_indices) for r in batch.reqs]
    extend_lens = [r.extend_input_len for r in batch.reqs]
    bs = len(batch.reqs)

    # 2. 直接在 pinned buffer 上填充数据
    # seq_lens
    torch.as_tensor(seq_lens, dtype=torch.int64, out=self.pinned_seq_lens[:bs])
    batch.seq_lens = self.pinned_seq_lens[:bs].to(self.device, non_blocking=True)
    batch.seq_lens_cpu = self.pinned_seq_lens[:bs].clone()  # CPU 副本

    # prefix_lens
    torch.as_tensor(prefix_lens, dtype=torch.int64, out=self.pinned_prefix_lens[:bs])
    prefix_lens_device = self.pinned_prefix_lens[:bs].to(self.device, non_blocking=True)

    # extend_lens
    torch.as_tensor(extend_lens, dtype=torch.int64, out=self.pinned_extend_lens[:bs])
    extend_lens_device = self.pinned_extend_lens[:bs].to(self.device, non_blocking=True)

    # extend_ids (flatten)
    extend_ids_flat = [token_id for ids in extend_ids for token_id in ids]
    num_tokens = len(extend_ids_flat)
    torch.as_tensor(extend_ids_flat, dtype=torch.int64, out=self.pinned_input_ids[:num_tokens])
    extend_ids_tensor = self.pinned_input_ids[:num_tokens].to(self.device, non_blocking=True)

    # req_pool_indices
    req_pool_indices = self.request_pool.alloc(bs)
    torch.as_tensor(req_pool_indices, dtype=torch.int64, out=self.pinned_req_pool_indices[:bs])
    req_pool_indices_tensor = self.pinned_req_pool_indices[:bs].to(self.device, non_blocking=True)

    # ... 其余逻辑保持不变
```

### 3. 修改 ForwardBatch.init_new 使用 pinned buffer

```python
@dataclass
class ForwardBatch:
    # 类级别的 pinned buffer (单例模式)
    _pinned_temperatures: torch.Tensor = None
    _pinned_top_ps: torch.Tensor = None
    _pinned_top_ks: torch.Tensor = None

    @classmethod
    def _ensure_pinned_buffers(cls, max_bs: int):
        if cls._pinned_temperatures is None:
            cls._pinned_temperatures = torch.empty(max_bs, dtype=torch.float32, pin_memory=True)
            cls._pinned_top_ps = torch.empty(max_bs, dtype=torch.float32, pin_memory=True)
            cls._pinned_top_ks = torch.empty(max_bs, dtype=torch.int64, pin_memory=True)

    @classmethod
    def init_new(cls, batch: ScheduledBatch, attn_backend):
        bs = len(batch.reqs)
        cls._ensure_pinned_buffers(max_bs=256)  # 或从 config 获取

        # 提取 sampling params 到 pinned buffer
        temps = [r.sampling_params.temperature for r in batch.reqs]
        top_ps = [r.sampling_params.top_p for r in batch.reqs]
        top_ks = [r.sampling_params.top_k for r in batch.reqs]

        torch.as_tensor(temps, dtype=torch.float32, out=cls._pinned_temperatures[:bs])
        torch.as_tensor(top_ps, dtype=torch.float32, out=cls._pinned_top_ps[:bs])
        torch.as_tensor(top_ks, dtype=torch.int64, out=cls._pinned_top_ks[:bs])

        dev = batch.device
        sampling_temperatures = cls._pinned_temperatures[:bs].to(dev, non_blocking=True)
        sampling_top_ps = cls._pinned_top_ps[:bs].to(dev, non_blocking=True)
        sampling_top_ks = cls._pinned_top_ks[:bs].to(dev, non_blocking=True)

        # ... 其余逻辑
```

## 预期效果

- **CPU → GPU 传输**: 真正的零拷贝异步传输，无内部临时 buffer 复制
- **内存分配**: 消除每次 step 的动态内存分配开销
- **forward_batch_init 时间**: 预计再减少 20-30%
- **更好的 CPU-GPU overlap**: 异步传输更快完成，GPU 可以更早开始计算

## 注意事项

1. **Buffer 大小**: 需要根据 `max_num_seqs` 和 `max_total_tokens` 合理配置
2. **内存占用**: Pinned memory 占用系统物理内存，不可 swap。需控制总量
3. **线程安全**: 如果引入多线程，需要为每个线程分配独立的 buffer
4. **错误处理**: Buffer 溢出时需要优雅降级（使用临时 tensor）

## 实现优先级

1. ✅ **Phase 1**: 消除 CUDA 同步点（已完成）
2. 🔄 **Phase 2**: 预分配 pinned memory buffer pool（本方案）
3. 🔄 **Phase 3**: 后台线程预先构建元数据（见方案 2）
4. 🔄 **Phase 4**: 批量向量化操作优化元数据构建（见方案 3）
