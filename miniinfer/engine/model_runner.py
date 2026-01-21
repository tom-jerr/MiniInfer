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
from .scheduler_batch import ForwardBatch, ForwardMode
from loader.weight import load_hf_weight
from config.model.qwen2 import Qwen2Config
from config.model.base import PretrainedConfig
from models.fused_qwen2 import Qwen2Model
from miniinfer.layers.sample import Sampler, SamplingBatchInfo
from miniinfer.layers.attention_backend.flashattention_backend import FlashAttentionBackend
import logging

logger = logging.getLogger(__name__)

class ModelRunner:
    """
    单机版模型执行器

    简化版实现，每个 sequence 独立维护 KV cache
    """

    def __init__(self, config: EngineConfig, kv_cache_mgr: Any, rank: int = 0, events: list = None, attn_backend: str="flash_attn"):
        self.config = config
        self.kv_cache_mgr = kv_cache_mgr
        self.rank = rank
        self.events = events or []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if "Qwen2" in config.model:
            self.model_config: PretrainedConfig = Qwen2Config.from_pretrained(
                config.model
            )
            self.model = Qwen2Model(self.model_config, device=self.device)
            logger.info("Initialized Qwen2Model")
        else:
            self.model = None
        self.sampler = Sampler()
        self.init_load_model()
        self.init_attn_backend(attn_backend)

    def init_load_model(self):
        """初始化加载模型"""
        if self.model is None:
            raise RuntimeError("Model is not initialized")
        self.state_dicts = load_hf_weight(self.config.model, self.device)
        self.model.load_weights(self.state_dicts)
        self.model.eval()
        logger.info(f"Model loaded on device {self.device}.")

    def init_attn_backend(self, attn_backend: str):
        if attn_backend not in ["flash_attn"]:
            raise ValueError(f"Unsupported attention backend: {attn_backend}")
        self.attn_backend = FlashAttentionBackend(self.kv_cache_mgr)
        logger.info(f"Using {attn_backend} as attention backend.")
    
    
    def forward(
        self,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")
    
    def forward_extend(
        self,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
       self.attn_backend.init_forward_metadata(forward_batch)
       return self.model.forward(
           forward_batch.input_ids,
           forward_batch.positions,
           forward_batch,
       )

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
       # TODO: support CUDA Graph
       self.attn_backend.init_forward_metadata(forward_batch)
       return self.model.forward(
           forward_batch.input_ids,
           forward_batch.positions,
           forward_batch,
    )

    def sample(
        self,
        logits: torch.Tensor,
        batch: ForwardBatch,
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

        next_token_ids = self.sampler(logits, sampling_batch_info)  # [num_seqs, 1]
        return next_token_ids
