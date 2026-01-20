"""
Scheduler - 单机版调度器

负责:
1. 管理请求队列 (waiting, running, finished)
2. 组织 batch (prefill-only / decode-only / mixed)
3. 调用 KVCacheManager 分配 KV Cache
"""

from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import torch

from .seqeunce import Sequence, SequenceStatus
from .engine_config import Config


class BatchType(Enum):
    """Batch 类型"""

    PREFILL_ONLY = auto()
    DECODE_ONLY = auto()
    MIXED = auto()


@dataclass
class ScheduledBatch:
    """调度后的 Batch"""

    batch_type: BatchType = BatchType.MIXED
    sequences: List[Sequence] = field(default_factory=list)
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs: List[Sequence] = field(default_factory=list)

    # Batch 输入数据
    input_ids: Optional[torch.Tensor] = None  # [total_tokens]
    position_ids: Optional[torch.Tensor] = None  # [total_tokens]

    # 用于区分 prefill 和 decode
    prefill_lens: List[int] = field(default_factory=list)
    decode_lens: List[int] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.sequences) == 0

    @property
    def num_prefill_tokens(self) -> int:
        return sum(self.prefill_lens)

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode_seqs)


class Scheduler:
    """
    单机版调度器

    简化版实现，不使用 KVCacheManager，直接使用 per-request cache
    """

    def __init__(self, config: Config):
        self.config = config
        self.max_batch_size = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens

        # 请求队列
        self.waiting_queue: List[Sequence] = []
        self.running_queue: List[Sequence] = []
        self.finished_queue: List[Sequence] = []

        # 请求 ID 映射
        self.seq_map: Dict[int, Sequence] = {}

    def add(self, seq: Sequence):
        """添加新请求到等待队列"""
        self.waiting_queue.append(seq)
        self.seq_map[seq.seq_id] = seq

    def has_unfinished(self) -> bool:
        """检查是否有未完成的请求"""
        return len(self.waiting_queue) > 0 or len(self.running_queue) > 0

    def get_num_unfinished(self) -> int:
        """获取未完成请求数量"""
        return len(self.waiting_queue) + len(self.running_queue)

    def schedule(self) -> ScheduledBatch:
        """
        调度一个 batch

        策略:
        1. 优先处理 running queue 中的 decode 请求
        2. 然后从 waiting queue 添加 prefill 请求
        """
        batch = ScheduledBatch()

        # 1. 添加 decode 请求 (running queue)
        for seq in self.running_queue:
            if len(batch.sequences) >= self.max_batch_size:
                break
            batch.decode_seqs.append(seq)
            batch.sequences.append(seq)
            batch.decode_lens.append(1)  # decode 每次只生成 1 个 token

        # 2. 添加 prefill 请求 (waiting queue)
        prefill_budget = self.max_batch_size - len(batch.sequences)
        token_budget = self.max_num_batched_tokens - batch.num_decode_tokens

        new_waiting = []
        for seq in self.waiting_queue:
            if len(batch.prefill_seqs) >= prefill_budget:
                new_waiting.append(seq)
                continue

            prefill_len = seq.num_tokens - seq.num_cached_tokens
            if batch.num_prefill_tokens + prefill_len > token_budget:
                new_waiting.append(seq)
                continue

            batch.prefill_seqs.append(seq)
            batch.sequences.append(seq)
            batch.prefill_lens.append(prefill_len)
            seq.status = SequenceStatus.RUNNING

        self.waiting_queue = new_waiting

        # 确定 batch 类型
        if batch.prefill_seqs and not batch.decode_seqs:
            batch.batch_type = BatchType.PREFILL_ONLY
        elif batch.decode_seqs and not batch.prefill_seqs:
            batch.batch_type = BatchType.DECODE_ONLY
        else:
            batch.batch_type = BatchType.MIXED

        # 准备输入数据
        if not batch.is_empty:
            self._prepare_batch_inputs(batch)

        return batch

    def _prepare_batch_inputs(self, batch: ScheduledBatch):
        """准备 batch 的输入 tensor"""
        all_input_ids = []
        all_position_ids = []

        # Prefill 请求: 输入完整的 prompt tokens
        for seq in batch.prefill_seqs:
            start_pos = seq.num_cached_tokens
            input_ids = seq.token_ids[start_pos:]
            positions = list(range(start_pos, seq.num_tokens))

            all_input_ids.extend(input_ids)
            all_position_ids.extend(positions)

        # Decode 请求: 只输入最后一个 token
        for seq in batch.decode_seqs:
            all_input_ids.append(seq.last_token)
            all_position_ids.append(seq.num_tokens - 1)

        if all_input_ids:
            batch.input_ids = torch.tensor(all_input_ids, dtype=torch.long)
            batch.position_ids = torch.tensor(all_position_ids, dtype=torch.long)

    def update_after_step(
        self,
        batch: ScheduledBatch,
        next_tokens: List[int],
        eos_token_id: int,
    ) -> List[Tuple[int, List[int]]]:
        """
        模型推理后更新状态

        Args:
            batch: 当前 batch
            next_tokens: 生成的 token 列表
            eos_token_id: EOS token ID

        Returns:
            完成的请求列表 [(seq_id, token_ids), ...]
        """
        finished_outputs = []

        # 分配 token 到对应的 sequence
        token_idx = 0

        # Prefill 请求: 移到 running queue
        for seq in batch.prefill_seqs:
            next_token = next_tokens[token_idx]
            token_idx += 1

            seq.append_token(next_token)
            seq.num_cached_tokens = seq.num_tokens - 1  # 标记已缓存

            # 检查是否完成
            if self._check_finished(seq, next_token, eos_token_id):
                seq.status = SequenceStatus.FINISHED
                self.finished_queue.append(seq)
                finished_outputs.append((seq.seq_id, seq.token_ids))
            else:
                self.running_queue.append(seq)

        # Decode 请求: 更新并检查完成
        new_running = []
        for seq in batch.decode_seqs:
            next_token = next_tokens[token_idx]
            token_idx += 1

            seq.append_token(next_token)
            seq.num_cached_tokens = seq.num_tokens - 1

            if self._check_finished(seq, next_token, eos_token_id):
                seq.status = SequenceStatus.FINISHED
                self.finished_queue.append(seq)
                finished_outputs.append((seq.seq_id, seq.token_ids))
                # 从 running queue 移除
                self.running_queue.remove(seq)
            # 否则保留在 running_queue

        return finished_outputs

    def _check_finished(self, seq: Sequence, token: int, eos_token_id: int) -> bool:
        """检查请求是否完成"""
        # 达到最大长度
        if seq.num_completion_tokens >= seq.max_tokens:
            return True

        # 遇到 EOS (除非设置了 ignore_eos)
        if token == eos_token_id and not seq.ignore_eos:
            return True

        return False

    def get_finished(self) -> List[Sequence]:
        """获取所有已完成的请求"""
        return self.finished_queue.copy()

    def clear_finished(self):
        """清空已完成队列"""
        self.finished_queue.clear()
