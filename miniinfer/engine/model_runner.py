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
from config.model.qwen3 import Qwen3Config
from config.model.base import PretrainedConfig
from models.fused_qwen2 import Qwen2ForCausalLM
from models.fused_qwen3 import Qwen3ForCausalLM
from miniinfer.layers.sample import Sampler, SamplingBatchInfo
from miniinfer.layers.attention_backend.flashattention_backend import (
  FlashAttention2Backend,
)
from miniinfer.engine.cuda_graph_runner import CudaGraphRunner
from miniinfer.utils.profiler_utils import profile_methods
import logging

logger = logging.getLogger(__name__)


@profile_methods("ModelRunner")
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

    model_type = getattr(config.hf_config, "model_type", "").lower()
    if model_type == "qwen2" or ("Qwen2" in config.model and model_type not in ("qwen3",)):
      self.model_config: PretrainedConfig = Qwen2Config.from_pretrained(config.model)
      self.model = Qwen2ForCausalLM(self.model_config, device=self.device)
      logger.info("Initialized Qwen2Model")
    elif model_type == "qwen3":
      self.model_config: PretrainedConfig = Qwen3Config.from_pretrained(config.model)
      self.model = Qwen3ForCausalLM(self.model_config, device=self.device)
      logger.info("Initialized Qwen3Model")
    else:
      self.model = None
    self.sampler = Sampler()
    self.init_load_model()

    # CUDA Graph support
    self.cuda_graph_runner: CudaGraphRunner = None
    self.use_cuda_graph: bool = not config.enforce_eager

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
      return self.forward_decode(forward_batch, return_hidden_states=return_hidden_states)
    elif forward_batch.forward_mode.is_extend():
      return self.forward_extend(forward_batch, return_hidden_states=return_hidden_states)
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
    # Use CUDA Graph if available and enabled
    if (
      self.use_cuda_graph
      and self.cuda_graph_runner is not None
      and self.cuda_graph_runner.is_available()
      and not return_hidden_states  # CUDA Graph doesn't support hidden states
    ):
      return self.cuda_graph_runner.replay(forward_batch)

    # Fallback to eager mode
    self.attn_backend.init_forward_metadata(forward_batch)
    return self.model.forward(
      forward_batch.input_ids,
      forward_batch.positions,
      forward_batch,
      return_hidden_states=return_hidden_states,
    )

  def init_cuda_graph(
    self,
    max_batch_size: int = 256,
    max_context_len: int = 4096,
  ) -> None:
    """
    Initialize CUDA Graph runner and warmup.

    Should be called after model and KV cache are fully initialized.
    This captures CUDA graphs for all power-of-2 batch sizes from 1 to max_batch_size.

    Args:
        max_batch_size: Maximum batch size to capture (typically max_num_seqs)
        max_context_len: Maximum context length for block table sizing
    """
    if not self.use_cuda_graph:
      logger.info("CUDA Graph disabled (enforce_eager=True)")
      return

    if self.attn_backend is None:
      raise RuntimeError(
        "Attention backend not initialized. "
        "Call set_kv_cache_mgr() first before init_cuda_graph()."
      )

    if self.kv_cache_mgr is None:
      raise RuntimeError(
        "KV cache manager not initialized. "
        "Call set_kv_cache_mgr() first before init_cuda_graph()."
      )

    logger.info(
      f"Initializing CUDA Graph runner with max_bs={max_batch_size}, "
      f"max_context_len={max_context_len}"
    )

    self.cuda_graph_runner = CudaGraphRunner(
      model=self.model,
      attn_backend=self.attn_backend,
      kv_cache_mgr=self.kv_cache_mgr,
      max_batch_size=max_batch_size,
      max_context_len=max_context_len,
      page_size=self.kv_cache_mgr.page_size,
      vocab_size=self.model_config.vocab_size,
      dtype=self.config.dtype,
      device=self.device,
    )

    # Warmup: capture all graphs
    self.cuda_graph_runner.warmup()
    logger.info("CUDA Graph warmup complete")

  def sample(
    self,
    logits: torch.Tensor,
    batch: ForwardBatch,
  ) -> torch.Tensor:
    """
    采样生成 token

    Args:
        logits: 通常为 [batch_size, vocab_size]（decode 以及优化后的 extend/mixed）。
                旧路径下 extend 可能为 [total_tokens, vocab_size]（所有序列 token 拼接）。
        batch: 当前 batch

    Returns:
        生成的 token 列表
    """
    if logits is None:
      return []

    # 在 extend（prefill/mixed）模式下，如果 logits 仍是 [total_tokens, vocab_size]，
    # 需要提取每个序列最后一个 token 的 logits；否则 logits 已经是 [batch_size, vocab_size]。
    if batch.forward_mode.is_extend():
      if logits.size(0) == batch.batch_size:
        logits_for_sampling = logits
      else:
        extend_lens = batch.extend_seq_lens_cpu or []
        last_token_indices = []
        cumsum = 0
        for length in extend_lens:
          last_token_indices.append(cumsum + int(length) - 1)
          cumsum += int(length)
        last_token_indices = torch.tensor(
          last_token_indices, device=logits.device, dtype=torch.int64
        )
        logits_for_sampling = logits[last_token_indices]
    else:
      # decode 模式下，每个序列只有一个 token，logits 已经是 [batch_size, vocab_size]
      logits_for_sampling = logits

    # Use pre-built sampling params from ForwardBatch to avoid per-step H2D
    temps = getattr(batch, "sampling_temperatures", None)
    top_ps = getattr(batch, "sampling_top_ps", None)
    top_ks = getattr(batch, "sampling_top_ks", None)
    if temps is not None and top_ps is not None and top_ks is not None:
      sampling_batch_info = SamplingBatchInfo(
        temperature=temps,
        top_ps=top_ps,
        top_ks=top_ks,
        max_top_k=getattr(batch, "sampling_max_top_k", None),
        vocab_size=logits.size(-1),
      )
    else:
      # Fallback: create from Python lists (legacy path)
      if batch.all_seqs is None:
        raise ValueError(
          "ForwardBatch is missing sampling tensors and `all_seqs`; "
          "cannot construct per-sequence SamplingBatchInfo."
        )
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
        max_top_k=max(
          1,
          max(
            (int(seq.sampling_params.top_k) for seq in batch.all_seqs),
            default=0,
          ),
        ),
        vocab_size=logits.size(-1),
      )

    next_token_ids = self.sampler(logits_for_sampling, sampling_batch_info)  # [num_seqs,]
    return next_token_ids
