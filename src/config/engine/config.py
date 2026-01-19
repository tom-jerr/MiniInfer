import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch


@dataclass
class EngineConfig:
    model: str
    max_num_seqs: int = 512
    max_context_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_context_len = min(
            self.max_context_len, self.hf_config.max_position_embeddings
        )
