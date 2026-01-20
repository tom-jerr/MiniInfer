"""
ModelRunner - 模型推理执行器

负责:
1. 加载模型
2. 执行 forward pass
3. 采样生成 token
"""

from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from config.engine.config import EngineConfig
from .scheduler import ScheduledBatch, BatchType
from ..engine.kv_cache import TinyKvFullCache
from loader.weight import load_hf_weight
from config.model.qwen2 import Qwen2Config
from config.model.base import PretrainedConfig
from models.fused_qwen2 import Qwen2Model
from layers.sample import Sampler


class ModelRunner:
    """
    单机版模型执行器

    简化版实现，每个 sequence 独立维护 KV cache
    """

    def __init__(self, config: EngineConfig, rank: int = 0, events: list = None):
        self.config = config
        self.rank = rank
        self.events = events or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if config.model is "Qwen2":
            self.model_config: PretrainedConfig = Qwen2Config.from_pretrained(
                config.model
            )
            self.model = Qwen2Model(self.model_config, device=self.device)
        else:
            self.model = None
        self.sampler = Sampler()
        self.init_load_model()

    def init_load_model(self):
        """初始化加载模型"""
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.state_dicts = load_hf_weight(self.config.model, self.device)
        self.model.load_weights(self.state_dicts)
        self.model.eval()
        print("Model loaded successfully.")

    def forward(
        self,
        batch: ScheduledBatch,
    ) -> torch.Tensor:
        """
        执行模型前向传播

        Args:
            batch: 调度后的 batch

        Returns:
            logits tensor [num_sequences, vocab_size]
        """
        if batch.is_empty:
            return None

        # 简化实现: 逐个 sequence 处理
        # 生产环境应该使用 batched attention
        all_logits = []

        with torch.no_grad():
            # 处理 prefill 请求
            for seq in batch.prefill_seqs:
                logits = self._forward_prefill(seq)
                all_logits.append(logits)

            # 处理 decode 请求
            for seq in batch.decode_seqs:
                logits = self._forward_decode(seq)
                all_logits.append(logits)

        if all_logits:
            return torch.stack(all_logits, dim=0)  # [num_seqs, vocab_size]
        return None

    def _forward_prefill(self, seq) -> torch.Tensor:
        """Prefill 阶段前向传播"""
        # 初始化 KV cache
        if seq.seq_id not in self.kv_caches:
            num_layers = self.config.hf_config.num_hidden_layers
            self.kv_caches[seq.seq_id] = [TinyKvFullCache() for _ in range(num_layers)]

        # 准备输入
        input_ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
        kv_cache = self.kv_caches[seq.seq_id]

        # 前向传播
        logits, updated_cache = self.model(
            input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            mask="causal",
        )

        # 更新 cache
        self.kv_caches[seq.seq_id] = updated_cache

        # 返回最后一个 token 的 logits
        return logits[0, -1, :]  # [vocab_size]

    def _forward_decode(self, seq) -> torch.Tensor:
        """Decode 阶段前向传播"""
        # 获取 KV cache
        kv_cache = self.kv_caches.get(seq.seq_id)
        if kv_cache is None:
            raise RuntimeError(f"KV cache not found for seq {seq.seq_id}")

        # 只输入最后一个 token
        input_ids = torch.tensor(
            [[seq.last_token]], dtype=torch.long, device=self.device
        )

        # 前向传播
        logits, updated_cache = self.model(
            input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            mask="causal",
        )

        # 更新 cache
        self.kv_caches[seq.seq_id] = updated_cache

        return logits[0, -1, :]  # [vocab_size]

    def sample(
        self,
        logits: torch.Tensor,
        batch: ScheduledBatch,
    ) -> torch.Tensor:
        """
        采样生成 token

        Args:
            logits: [num_seqs, vocab_size]
            batch: 当前 batch

        Returns:
            生成的 token 列表
        """
        if logits is None:
            return []

        sampling_batch_info = Sampler.SamplingBatchInfo(
            temperature=torch.tensor(
                [seq.sampling_params.temperature for seq in batch.all_seqs],
                device=logits.device,
            ),
            top_ps=torch.tensor(
                [seq.sampling_params.top_p for seq in batch.all_seqs],
                device=logits.device,
            ),
            top_ks=torch.tensor(
                [seq.sampling_params.top_k for seq in batch.all_seqs],
                device=logits.device,
            ),
            vocab_size=logits.size(-1),
        )

        next_token_ids = self.sampler(logits, sampling_batch_info)  # [num_seqs, 1]
        return next_token_ids
