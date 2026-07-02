from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoConfig
import torch


@dataclass
class EngineConfig:
  """
  推理引擎配置

  显存管理相关参数:
  - gpu_memory_utilization: GPU 显存利用率 (0.0 ~ 1.0)，用于计算 KV cache 可用容量
  - max_total_tokens: 手动指定 KV cache 可容纳的最大 token 数（覆盖自动计算）
  - max_num_batched_tokens: 单次推理的最大 token 数（用于 prefill 调度）
  - num_pages: 手动指定 KV cache 页面数（覆盖自动计算）
  - page_size: 每页包含的 token 数
  """

  model_path: str
  dtype: torch.dtype = torch.float16
  # 默认 flashinfer：无 flash_attn 的 page_size-必须被 256 整除约束，且是 vLLM-grade
  # paged decode attention。如需切回 FA2，传 attention_backend="flash_attn"。
  # 注：benchmark 显示二者在 page_size=256 下吞吐接近（见
  # docs/性能优化_FA2对比_与长序列attention瓶颈.md §8）。
  attention_backend: str = "flashinfer"

  # ============ 显存管理配置 ============
  # GPU 显存利用率，用于自动计算 KV cache 容量
  # 默认 0.6 更保守，避免 OOM。生产环境可根据实际情况调整到 0.7-0.8
  gpu_memory_utilization: float = 0.6
  # 每页 token 数
  page_size: int = 256

  # ============ 调度配置 ============
  max_num_seqs: int = 256
  max_context_len: int = 4096
  max_extend_len: int = 8192  # for chunked prefill
  chunked_prefill_size: int = 4096
  enable_prefix_cache: bool = True
  enable_chunked_prefill: bool = False

  # ============ 并行配置 ============
  tensor_parallel_size: int = 1

  # ============ 其他配置 ============
  enforce_eager: bool = False
  hf_config: Optional[AutoConfig] = field(default=None, repr=False)
  eos: int = -1

  # ============ CUDA Graph 配置 ============
  # 是否启用 CUDA Graph 加速 decode 阶段 (enforce_eager=True 时禁用)
  # cuda_graph_max_bs: 0 表示使用 max_num_seqs
  cuda_graph_max_bs: int = 0

  # ============ Overlap 配置 ============
  # 是否启用双 Batch 交替执行 (overlap scheduling)
  # 启用后 GPU 执行当前 batch 时 CPU 并行处理上一 batch
  enable_overlap: bool = True

  # 别名支持（兼容旧配置）
  @property
  def model(self) -> str:
    return self.model_path

  @property
  def memory_ratio(self) -> float:
    """兼容旧的 memory_ratio 参数名"""
    return self.gpu_memory_utilization

  def __post_init__(self):
    # 加载 HuggingFace 配置
    self.hf_config = AutoConfig.from_pretrained(self.model_path)

    # 限制 max_context_len 不超过模型支持的最大位置编码
    self.max_context_len = min(self.max_context_len, self.hf_config.max_position_embeddings)
