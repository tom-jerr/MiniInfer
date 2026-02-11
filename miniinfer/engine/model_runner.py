"""
ModelRunner - 模型推理执行器

负责:
1. 加载模型
2. 执行 forward pass
3. 采样生成 token
"""

from typing import List, Optional, Tuple, Any
from models.base import BaseModelOutput
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from config.engine.config import EngineConfig
from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode
from loader.weight import load_hf_weight
from config.model.qwen2 import Qwen2Config
from config.model.base import PretrainedConfig
from models.fused_qwen2 import Qwen2ForCausalLM
from miniinfer.layers.sample import Sampler, SamplingBatchInfo
from miniinfer.layers.attention_backend.flashattention_backend import (
    FlashAttention2Backend,
)
import logging

logger = logging.getLogger(__name__)


class ModelRunner:
    """
    单机版模型执行器

    简化版实现，每个 sequence 独立维护 KV cache
    """

    def __init__(
        self,
        config: EngineConfig,
        kv_cache_mgr: Any = None,
        rank: int = 0,
        events: list = None,
        attn_backend: str = "flash_attn",
    ):
        self.config = config
        self.kv_cache_mgr = kv_cache_mgr
        self.rank = rank
        self.events = events or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attn_backend = None
        self._attn_backend_type = attn_backend

        if "Qwen2" in config.model:
            self.model_config: PretrainedConfig = Qwen2Config.from_pretrained(
                config.model
            )
            self.model = Qwen2ForCausalLM(self.model_config, device=self.device)
            logger.info("Initialized Qwen2Model")
        else:
            self.model = None
        self.sampler = Sampler()
        self.init_load_model()

        # 如果提供了 kv_cache_mgr，立即初始化 attn_backend
        if kv_cache_mgr is not None:
            self.init_attn_backend(attn_backend)

    def init_load_model(self):
        """初始化加载模型"""
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.state_dicts = load_hf_weight(self.config.model, self.device)
        self.model.load_weights(self.state_dicts)
        self.model.eval()
        logger.info(f"Model loaded on device {self.device}.")

    def init_attn_backend(self, attn_backend: str = None):
        """初始化 attention backend（需要先提供 kv_cache_mgr）"""
        if self.kv_cache_mgr is None:
            raise RuntimeError("Cannot initialize attn_backend without kv_cache_mgr")

        attn_backend = attn_backend or self._attn_backend_type
        if attn_backend not in ["flash_attn"]:
            raise ValueError(f"Unsupported attention backend: {attn_backend}")
        self.attn_backend = FlashAttention2Backend(self.kv_cache_mgr)
        logger.info(f"Using {self.attn_backend.type()} as attention backend.")

    def set_kv_cache_mgr(self, kv_cache_mgr: Any):
        """设置 KV Cache 管理器并初始化 attn_backend"""
        self.kv_cache_mgr = kv_cache_mgr
        self.init_attn_backend()

    def forward(
        self,
        forward_batch: ForwardBatch,
        return_hidden_states: bool = False,
    ) -> BaseModelOutput:
        if self.attn_backend is None:
            raise RuntimeError(
                "Attention backend not initialized. "
                "Call set_kv_cache_mgr() first to initialize the backend."
            )

        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                forward_batch, return_hidden_states=return_hidden_states
            )
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(
                forward_batch, return_hidden_states=return_hidden_states
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        return_hidden_states: bool = False,
    ) -> BaseModelOutput:
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            return_hidden_states=return_hidden_states,
        )

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        return_hidden_states: bool = False,
    ) -> BaseModelOutput:
        # TODO: support CUDA Graph
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            return_hidden_states=return_hidden_states,
        )

    def sample(
        self,
        logits: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        """
        采样生成 token

        Args:
            logits: 在 decode 模式下为 [batch_size, vocab_size]，
                    在 extend 模式下为 [total_tokens, vocab_size]（所有序列 token 拼接）
            batch: 当前 batch

        Returns:
            生成的 token 列表
        """
        if logits is None:
            return []

        # 在 extend（prefill）模式下，需要从 logits 中提取每个序列最后一个 token 的 logits
        # logits 的形状是 [total_tokens, vocab_size]，其中 total_tokens = sum(extend_lens)
        # 我们只需要每个序列最后一个位置的 logits 用于采样
        if batch.forward_mode.is_extend():
            # 计算每个序列最后一个 token 在 logits 中的索引
            extend_lens = batch.extend_seq_lens_cpu
            last_token_indices = []
            cumsum = 0
            for length in extend_lens:
                last_token_indices.append(cumsum + length - 1)
                cumsum += length
            last_token_indices = torch.tensor(last_token_indices, device=logits.device)
            logits_for_sampling = logits[last_token_indices]
        else:
            # decode 模式下，每个序列只有一个 token，logits 已经是 [batch_size, vocab_size]
            logits_for_sampling = logits

        sampling_batch_info = SamplingBatchInfo(
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

        next_token_ids = self.sampler(
            logits_for_sampling, sampling_batch_info
        )  # [num_seqs,]
        return next_token_ids
