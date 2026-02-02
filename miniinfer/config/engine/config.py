import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch


@dataclass
class EngineConfig:
    model: str
    max_num_seqs: int = 512
    max_context_len: int = 4096
    max_total_tokens: int = 20480
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1

    # ============ Chunked Prefill / Memory Budget 配置 ============
    # 每个 batch 的最大 token 数（包括 extend + decode）
    # 设置为 None 则使用 max_total_tokens
    max_num_batched_tokens: int | None = None
    # 是否启用 chunked prefill（将长 prefill 拆分成多个 chunk）
    enable_chunked_prefill: bool = False
    # chunked prefill 的 chunk 大小（每次最多处理多少 token）
    chunked_prefill_size: int = 4096
    # 为 decode 请求预留的 token 数（防止 prefill 饿死 decode）
    reserved_decode_tokens: int = 256

    def __post_init__(self):
        # assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_context_len = min(
            self.max_context_len, self.hf_config.max_position_embeddings
        )

        # 如果没有指定 max_num_batched_tokens，使用 max_total_tokens
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = self.max_total_tokens
