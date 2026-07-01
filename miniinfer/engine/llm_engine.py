"""
LLMEngine - 对外暴露的统一接口

运行模式:
1. 单进程模式 (默认): 所有组件在同一进程内运行
2. 多进程模式: 预留接口（当前未实现）

Usage:
    # 单进程模式 (默认)
    engine = LLMEngine(model_path="...")
    outputs = engine.generate(prompts, sampling_params)

    # 多进程模式 (当前未实现，会抛 NotImplementedError)
    # engine = LLMEngine(model_path="...", use_multiprocess=True)
"""

import atexit
from miniinfer.utils import get_logger
import queue
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import List, Optional, Union, Dict, Any, Generator, Sequence
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch

from miniinfer.kvcache.kv_cache_manager import KVCacheManager
from miniinfer.config.engine.config import EngineConfig
from miniinfer.utils.memory_utils import calc_max_total_tokens
from miniinfer.utils.profiler_utils import profile_methods, stage
from .model_runner import ModelRunner
from .overlap_executor import OverlapExecutor
from .incremental_batch_tokenizer import IncrementalBatchTokenizer
from miniinfer.scheduler.scheduler import Scheduler
from miniinfer.scheduler.scheduler_batch import (
  Req,
  ScheduledBatch,
  ForwardBatch,
  BatchResult,
)

from ..utils.sampling_params import SamplingParams
from ..utils.crash_logger import log_inference_crash
from .detokenizer import IncrementalDecoder

logger = get_logger(__name__)


@dataclass
class RequestOutput:
  """单个请求的输出结果"""

  request_id: int
  delta_text: str
  full_text: str
  token_id: int
  output_token_ids: List[int]
  finished: bool
  finish_reason: Optional[str] = None


@dataclass
class StepOutput:
  """单步推理的输出"""

  outputs: List[RequestOutput]
  num_prefill_tokens: int = 0
  num_decode_tokens: int = 0
  # When this step runs a prefill/extend batch, expose per-request cache stats.
  prefill_prefix_lens: Dict[int, int] = field(default_factory=dict)
  prefill_extend_lens: Dict[int, int] = field(default_factory=dict)

  @property
  def has_output(self) -> bool:
    return len(self.outputs) > 0


@profile_methods("LLMEngine")
class LLMEngine:
  """
  LLM 推理引擎

  当前仅支持单进程模式。
  多进程模式接口保留，但暂未实现（use_multiprocess=True 会抛异常）。
  """

  def __init__(
    self,
    model: str,
    use_multiprocess: bool = False,
    **kwargs,
  ):
    """
    初始化 LLM Engine

    Args:
        model: 模型路径或 HuggingFace model id
        use_multiprocess: 是否使用多进程模式
        **kwargs: 其他配置参数
    """
    # 提取出 EngineConfig 类真正定义了的参数
    config_fields = {field.name for field in fields(EngineConfig)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    self.config = EngineConfig(model, **config_kwargs)

    # Prompt formatting (chat template) config.
    # - True: always use tokenizer.apply_chat_template when available
    # - False: never use chat template (raw text completion)
    # - "auto": enable for likely chat/instruct models
    self.use_chat_template = kwargs.get("use_chat_template", "auto")
    self.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
    self.debug = bool(kwargs.get("debug", True))
    self.debug_logit_topk = int(kwargs.get("debug_logit_topk", 5))
    # Optional callback to generate per-batch vocab masks for grammar-constrained sampling.
    # Signature: fn(forward_batch: ForwardBatch) -> Optional[torch.Tensor]
    self.vocab_mask_fn = kwargs.get("vocab_mask_fn", None)

    self.use_multiprocess = use_multiprocess
    self._started = False

    if use_multiprocess:
      raise NotImplementedError(
        "Multiprocess mode is not implemented yet. Please use use_multiprocess=False."
      )

    self._init_single_process_mode()
    self._req_by_id: Dict[int, Req] = {}

    atexit.register(self.stop)

  # ======================== private helper function ========================
  def _debug_print_topk_next_tokens(
    self, logits: torch.Tensor, forward_batch: ForwardBatch
  ) -> None:
    topk = self.debug_logit_topk
    if topk <= 0:
      return
    if logits is None:
      return

    with torch.no_grad():
      if forward_batch.forward_mode.is_extend():
        if logits.size(0) == forward_batch.batch_size:
          logits_for_sampling = logits
        else:
          extend_lens = forward_batch.extend_seq_lens_cpu or []
          last_token_indices: List[int] = []
          cumsum = 0
          for length in extend_lens:
            last_token_indices.append(cumsum + int(length) - 1)
            cumsum += int(length)
          if not last_token_indices:
            return
          # 优化：使用 torch.as_tensor 避免 CUDA 同步
          idx_cpu = torch.as_tensor(last_token_indices, dtype=torch.int64)
          idx = idx_cpu.to(logits.device, non_blocking=True)
          logits_for_sampling = logits[idx]
      else:
        logits_for_sampling = logits

      k = min(int(topk), int(logits_for_sampling.shape[-1]))
      top_vals, top_ids = torch.topk(logits_for_sampling, k=k, dim=-1)

      for i, req in enumerate(forward_batch.all_seqs):
        candidates = []
        for val, tid in zip(top_vals[i].tolist(), top_ids[i].tolist()):
          text = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
          candidates.append((int(tid), val, repr(text)))
        logger.debug("req=%s top%d next_token candidates: %s", req.req_id, k, candidates)

  # ======== prompt encoding (compat) ========
  def _encode_prompt(self, prompt: str) -> List[int]:
    """
    Encode a user prompt into token ids.

    Kept for backward-compat/tests; production code uses `self.prompt_tokenizer`.
    """
    prompt_tokenizer = getattr(self, "prompt_tokenizer", None)
    if prompt_tokenizer is None:
      prompt_tokenizer = IncrementalBatchTokenizer(
        self.tokenizer,
        use_chat_template=self.use_chat_template,
        model_name=str(getattr(self.config, "model", "")),
        system_prompt=self.system_prompt,
      )
    with stage("stage::Tokenizer.encode_one"):
      return prompt_tokenizer.encode_one(prompt)

  # ======== init helper function =========
  def _init_single_process_mode(self):
    """初始化单进程模式"""
    logger.info("Initializing LLMEngine in single-process mode")

    # TP 子进程（暂未启用）
    self.ps = []
    self.events = []
    # ======== engine config =============
    config = self.config

    # 步骤 1: 先创建 ModelRunner 并加载模型（不传入 kv_cache_mgr）
    logger.info("Step 1: Loading model...")
    with stage("stage::LLMEngine.init.model_runner"):
      self.model_runner = ModelRunner(
        self.config,
        kv_cache_mgr=None,
        rank=0,
        events=self.events,
        attn_backend=config.attention_backend,
      )

    # 步骤 2: 模型加载完成后，基于实际已分配显存计算 max_total_tokens
    logger.info("Step 2: Calculating max_total_tokens based on actual GPU memory usage...")
    with stage("stage::LLMEngine.init.calc_max_total_tokens"):
      self.max_total_tokens = calc_max_total_tokens(self.config, log=logger)

    # 步骤 3: 初始化 KV Cache 管理器
    logger.info("Step 3: Initializing KV Cache Manager...")
    with stage("stage::LLMEngine.init.kv_cache_mgr"):
      self.kv_cache_mgr = KVCacheManager(
        size=self.max_total_tokens,
        max_requests=config.max_num_seqs,
        max_context_len=config.max_context_len,
        num_layers=config.hf_config.num_hidden_layers,
        num_heads=config.hf_config.num_key_value_heads,  # 使用 KV head 数量，支持 GQA
        head_dim=getattr(
          config.hf_config,
          "head_dim",
          config.hf_config.hidden_size // config.hf_config.num_attention_heads,
        ),
        dtype=config.dtype,
        device="cuda",
        enable_prefix_cache=config.enable_prefix_cache,
        page_size=config.page_size,
        max_extend_tokens=config.max_extend_len,
      )

    # 步骤 4: 设置 ModelRunner 的 kv_cache_mgr 并初始化 attn_backend
    logger.info("Step 4: Initializing attention backend...")
    with stage("stage::LLMEngine.init.set_kv_cache_mgr"):
      self.model_runner.set_kv_cache_mgr(self.kv_cache_mgr)

    # Tokenizer
    with stage("stage::Tokenizer.from_pretrained"):
      self.tokenizer = AutoTokenizer.from_pretrained(
        self.config.model,
        use_fast=True,
        trust_remote_code=True,
      )
    with stage("stage::LLMEngine.init.prompt_tokenizer"):
      self.prompt_tokenizer = IncrementalBatchTokenizer(
        self.tokenizer,
        use_chat_template=self.use_chat_template,
        model_name=str(getattr(self.config, "model", "")),
        system_prompt=self.system_prompt,
      )
    # Stream detokenizer
    with stage("stage::LLMEngine.init.detokenizer"):
      self.detokenizer = IncrementalDecoder(
        tokenizer=self.tokenizer,
        skip_special_tokens=True,
      )
    self.config.eos = self.tokenizer.eos_token_id

    # Scheduler (传入 tokenizer 和 model_runner)
    with stage("stage::LLMEngine.init.scheduler"):
      self.scheduler = Scheduler(self.config, self.tokenizer, self.kv_cache_mgr)

    # 步骤 5: 初始化 CUDA Graph (如果未禁用)
    if not self.config.enforce_eager:
      logger.info("Step 5: Initializing CUDA Graph for decode acceleration...")
      cuda_graph_max_bs = self.config.cuda_graph_max_bs
      if cuda_graph_max_bs <= 0:
        cuda_graph_max_bs = self.config.max_num_seqs
      with stage("stage::LLMEngine.init.cuda_graph"):
        self.model_runner.init_cuda_graph(
          max_batch_size=cuda_graph_max_bs,
          max_context_len=self.config.max_context_len,
        )
    else:
      logger.info("Step 5: CUDA Graph disabled (enforce_eager=True)")

    # 步骤 6: 初始化 Overlap Executor (双 batch 交替执行)
    enable_overlap = self.config.enable_overlap
    logger.info(f"Step 6: Initializing Overlap Executor (enabled={enable_overlap})...")
    with stage("stage::LLMEngine.init.overlap_executor"):
      self.overlap_executor = OverlapExecutor(
        max_running_requests=self.config.max_num_seqs,
        max_chunks_per_request=4,
        device="cuda",
        enable_overlap=enable_overlap,
      )

    # 步骤 7: 预热 release path，避免第一次 drain_pending_releases() 触发
    # Runtime Triggered Module Loading。
    with stage("stage::LLMEngine.init.release_path_warmup"):
      self.kv_cache_mgr.warmup_release_path()

    # 步骤 8: 预热 schedule path，覆盖 prefill / decode / mixed 中首次触发的
    # CUDA/Triton 模块加载。
    with stage("stage::LLMEngine.init.schedule_path_warmup"):
      self.scheduler.warmup_schedule_path(device=torch.device("cuda"))

    # 追踪上一个 batch 用于 overlap 判断
    self._last_batch: Optional[ScheduledBatch] = None

    # 异步请求队列：支持在推理过程中动态添加请求
    # 线程安全，可从任意线程调用 submit_request()
    self._pending_requests: queue.Queue = queue.Queue()

    self._started = True

  # ======== add request and step function ========
  def add_requests(
    self,
    prompts: Sequence[Union[str, List[int]]],
    sampling_params: Sequence[SamplingParams],
  ) -> List[int]:
    if not prompts:
      return []
    if len(prompts) != len(sampling_params):
      raise ValueError(
        f"prompts and sampling_params length mismatch: {len(prompts)} vs {len(sampling_params)}"
      )

    token_ids_list: List[List[int]] = [[] for _ in range(len(prompts))]
    to_encode: List[str] = []
    to_encode_indices: List[int] = []

    for i, prompt in enumerate(prompts):
      if isinstance(prompt, str):
        to_encode.append(prompt)
        to_encode_indices.append(i)
      else:
        token_ids_list[i] = list(prompt)

    if to_encode:
      with stage("stage::Tokenizer.encode_batch"):
        encoded = self.prompt_tokenizer.encode_batch(to_encode)
      for i, ids in zip(to_encode_indices, encoded):
        token_ids_list[i] = ids

    request_ids: List[int] = []
    for token_ids, param in zip(token_ids_list, sampling_params):
      req = Req(token_ids, param)
      self.scheduler.add(req)
      self._req_by_id[req.req_id] = req
      request_ids.append(req.req_id)
    return request_ids

  def submit_request(
    self,
    prompt: Union[str, List[int]],
    sampling_params: Optional[SamplingParams] = None,
  ) -> int:
    """
    异步提交请求（线程安全）

    请求会被加入待处理队列，在下一次 step_overlap() 调用时处理。
    可以从任意线程调用此方法，适用于 online serving 场景。

    Args:
        prompt: 输入 prompt（文本或 token ids）
        sampling_params: 采样参数

    Returns:
        request_id: 请求 ID
    """
    if sampling_params is None:
      sampling_params = SamplingParams()

    # 预先 tokenize（如果需要）
    if isinstance(prompt, str):
      token_ids = self.prompt_tokenizer.encode_one(prompt)
    else:
      token_ids = list(prompt)

    req = Req(token_ids, sampling_params)
    self._req_by_id[req.req_id] = req
    self._pending_requests.put(req)
    return req.req_id

  def _recv_requests(self) -> int:
    """
    处理异步提交的请求（SGLang 风格 recv_requests）

    从 _pending_requests 队列中取出所有待处理请求，加入 scheduler。
    此方法在 step_overlap() 开头调用。

    Returns:
        处理的请求数量
    """
    count = 0
    while not self._pending_requests.empty():
      try:
        req = self._pending_requests.get_nowait()
        self.scheduler.add(req)
        count += 1
      except Exception:
        break
    return count

  def _is_disable_overlap_for_batch(self, batch: Optional[ScheduledBatch]) -> bool:
    """
    判断是否应该禁用当前 batch 的 overlap（SGLang 风格）

    禁用条件：
    1. overlap 未启用
    2. 连续两个 prefill/extend batch（优化 TTFT）
    3. 当前 batch 为空

    对于连续 prefill 禁用 overlap 的理由：
    - 假设 batch A 是 prefill，batch B 也是 prefill
    - 如果启用 overlap，process(A) 会在 forward(B) 期间执行
    - 但 batch A 的用户想尽快看到首个 token（TTFT）
    - 禁用 overlap 后，process(A) 在 forward(B) 之前完成
    - 这样 batch A 的首个 token 更早返回

    Args:
        batch: 当前要执行的 batch

    Returns:
        True 如果应该禁用 overlap
    """
    if not self.config.enable_overlap:
      return True

    if batch is None:
      return True

    # 连续两个 prefill/extend batch 禁用 overlap 以优化 TTFT
    if (
      self._last_batch is not None
      and self._last_batch.forward_mode.is_extend()
      and batch.forward_mode.is_extend()
    ):
      return True

    return False

  def _build_vocab_mask(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
    fn = getattr(self, "vocab_mask_fn", None)
    if fn is None:
      return None
    mask = fn(forward_batch)
    if mask is None:
      return None
    if not isinstance(mask, torch.Tensor):
      mask = torch.as_tensor(mask)
    if mask.dtype != torch.bool:
      mask = mask.to(torch.bool)
    return mask

  def step(self) -> StepOutput:
    """
    执行单步推理（非阻塞式）

    每次调用执行一个推理步骤：调度 -> forward -> sample -> 增量解码。
    新请求可以在任意时刻通过 add_request() 添加，不会阻塞此方法。

    Returns:
        StepOutput: 包含本步所有请求的增量输出
    """
    # Drain deferred releases first so radix/prefix cache state is up-to-date for scheduling.
    # This matters for cases like: req1 finishes -> req2 with same prompt is added immediately.
    if getattr(self.scheduler, "pending_release_reqs", None):
      self.scheduler.drain_pending_releases()

    # 1. 调度获取 batch
    try:
      with stage("stage::LLMEngine.step.schedule"):
        batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=False)
    except Exception as exc:
      log_inference_crash(
        exc, scheduler=self.scheduler, model_runner=self.model_runner, stage="schedule"
      )
      raise
    if batch is None or len(batch.reqs) == 0:
      # No work this round; make sure any deferred releases are flushed so memory isn't leaked.
      if getattr(self.scheduler, "pending_release_reqs", None):
        self.scheduler.drain_pending_releases()
      return StepOutput(outputs=[])

    # 2. 执行 forward (compute stream)
    forward_batch = None
    try:
      with stage("stage::LLMEngine.step.forward_batch_init"):
        forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)

      with stage("stage::LLMEngine.step.forward"):
        output = self.model_runner.forward(forward_batch)
        logits = output.logits
      # if self.debug_logit_topk > 0:
      # self._debug_print_topk_next_tokens(logits, forward_batch)
      # 3. 采样
      with stage("stage::LLMEngine.step.sample"):
        next_tokens = self.model_runner.sample(logits, forward_batch)
    except Exception as exc:
      log_inference_crash(
        exc,
        scheduler=self.scheduler,
        model_runner=self.model_runner,
        forward_batch=forward_batch,
        scheduled_batch=batch,
        stage="forward/sample",
      )
      raise

    # 4. 处理结果并增量解码
    with stage("stage::LLMEngine.step.process_result"):
      result = BatchResult(logits=logits, next_token_ids=next_tokens)
      outputs = self._process_step_result(batch, result)

    # 5. 统计 token 数
    num_prefill = 0
    num_decode = 0
    prefill_prefix_lens: Dict[int, int] = {}
    prefill_extend_lens: Dict[int, int] = {}

    decode_req_set = set()
    if batch.decoding_reqs:
      decode_req_set = {id(r) for r in batch.decoding_reqs}

    if batch.forward_mode.is_extend():
      # is_extend() covers both EXTEND and MIXED modes
      if batch.prefix_lens and batch.extend_lens:
        for req, pre_len, ext_len in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
          if id(req) in decode_req_set:
            # decode 请求在 mixed batch 中视为 decode token
            num_decode += 1
          else:
            num_prefill += ext_len
            prefill_prefix_lens[int(req.req_id)] = int(pre_len)
            prefill_extend_lens[int(req.req_id)] = int(ext_len)
      else:
        num_prefill = sum(batch.extend_lens) if batch.extend_lens else 0
    elif batch.forward_mode.is_decode():
      num_decode = len(batch.reqs)

    return StepOutput(
      outputs=outputs,
      num_prefill_tokens=num_prefill,
      num_decode_tokens=num_decode,
      prefill_prefix_lens=prefill_prefix_lens,
      prefill_extend_lens=prefill_extend_lens,
    )

  def step_overlap(self) -> StepOutput:
    """
    双 Batch 交替执行的推理步骤（SGLang 风格）

    核心设计（参考 SGLang event_loop_overlap）:
      - Placeholder 路径: schedule(N) → run_async(N) → process(N-1)
                                            ↑               ↑
                                      GPU 先启动    与 forward(N) 并行

      - Conservative 路径: process(N-1) → schedule(N) → run_async(N)
                              ↑
                     schedule 需要正确的 output_ids

    SGLang 的关键优化：
      1. 先启动 GPU，然后在 GPU 执行时处理上一批结果
      2. 连续两个 prefill batch 禁用 overlap，优化 TTFT（首 token 延迟）
      3. recv_requests 在循环开头处理异步提交的新请求

    Conservative 路径用于：首批请求、near-max-tokens 边界、连续 prefill。

    Phases:
      0. recv_requests: 处理 submit_request() 异步提交的请求
      1. 判断是否可以使用 placeholder (can_use_placeholder)
      2. Conservative: 先处理 N-1 (如果需要)
      3. Schedule: 获取下一个 batch
      3.5. 检测连续 prefill，禁用 overlap 优化 TTFT
      4. run_async: 启动 GPU forward
      5. Placeholder: GPU 运行时处理 N-1

    Returns:
        StepOutput: 包含本步所有请求的增量输出
    """
    outputs: List[RequestOutput] = []
    num_prefill = 0
    num_decode = 0
    prefill_prefix_lens: Dict[int, int] = {}
    prefill_extend_lens: Dict[int, int] = {}

    # ===================================================================
    # Phase 0: recv_requests — 处理异步提交的请求 (SGLang 风格)
    #
    # 从 _pending_requests 队列中取出所有待处理请求，加入 scheduler。
    # 这允许用户在推理过程中动态添加请求（通过 submit_request()）。
    # ===================================================================
    if getattr(self, "_pending_requests", None) is not None:
      with stage("stage::LLMEngine.step_overlap.recv_requests"):
        self._recv_requests()

    # ===================================================================
    # Phase 1: 判断路径
    #
    # can_use_placeholder 决定是否走 SGLang 风格 overlap：
    # - True: schedule 使用 placeholder input_ids，无需等待 N-1 处理完成
    # - False: schedule 需要正确的 output_ids，必须先处理 N-1
    # ===================================================================
    running_batch = self.scheduler.running_batch
    has_near_max_tokens_req = False
    if running_batch is not None and running_batch.reqs:
      for req in running_batch.reqs:
        if req.finished:
          continue
        max_tokens = int(getattr(req, "max_tokens", 0) or 0)
        if max_tokens > 0 and (len(getattr(req, "output_ids", [])) + 1) >= max_tokens:
          has_near_max_tokens_req = True
          break

    can_use_placeholder = (
      self.config.enable_overlap
      and not has_near_max_tokens_req
      and len(running_batch.reqs) > 0
      and running_batch.output_ids is not None
      # and len(running_batch.output_ids) == len(running_batch.reqs)
    )

    # ===================================================================
    # Phase 2: Conservative 路径 — 先处理 N-1
    #
    # 仅当无法使用 placeholder 时（首批/near-max-tokens），先同步处理上一批。
    # 这确保 schedule(N) 看到正确的 output_ids。
    # ===================================================================
    pending_step_out: Optional[StepOutput] = None

    if not can_use_placeholder and self.overlap_executor.has_pending():
      with stage("stage::LLMEngine.step_overlap.process_conservative"):
        try:
          pending_step_out = self.overlap_executor.process_pending_batch(self._process_overlap_result)
        except Exception as exc:
          log_inference_crash(
            exc, scheduler=self.scheduler, model_runner=self.model_runner,
            stage="step_overlap.process_conservative",
          )
          raise

    # ===================================================================
    # Phase 3: Schedule（在 schedule_stream 上执行 CPU→GPU 数据准备）
    #
    # 所有 schedule 和 prepare 操作在 schedule_stream 上执行，
    # forward_stream 会等待 schedule_stream 完成后再启动 GPU 计算。
    # ===================================================================
    schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
    # When CUDA graph is in use, schedule_stream's GPU prep (H2D copies, position
    # compute, allocations) for step N+1 must not race with forward_stream's graph
    # replay for step N. Make schedule_stream wait for the last forward_stream op
    # (the graph replay) before issuing its own GPU work. This is non-blocking on
    # the CPU (unlike forward_stream.synchronize()), preserving CPU post-processing
    # overlap with GPU forward.
    if schedule_stream is not None and self.model_runner.use_cuda_graph:
      schedule_stream.wait_stream(self.overlap_executor.forward_stream)
    try:
      if schedule_stream is not None:
        with torch.cuda.stream(schedule_stream):
          if can_use_placeholder:
            with stage("stage::LLMEngine.step_overlap.schedule_placeholder"):
              batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=True)
          else:
            with stage("stage::LLMEngine.step_overlap.schedule"):
              batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=False)
      else:
        if can_use_placeholder:
          with stage("stage::LLMEngine.step_overlap.schedule_placeholder"):
            batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=True)
        else:
          with stage("stage::LLMEngine.step_overlap.schedule"):
            batch = self.scheduler.schedule(self.model_runner.device, skip_decode_input_ids=False)
    except Exception as exc:
      log_inference_crash(
        exc, scheduler=self.scheduler, model_runner=self.model_runner, stage="step_overlap.schedule"
      )
      raise

    # ===================================================================
    # Phase 3.5: 检测连续 prefill，决定是否禁用 overlap (SGLang 风格)
    #
    # 连续两个 prefill/extend batch 会禁用 overlap 以优化 TTFT：
    # - 第一个 prefill batch 的用户希望尽快看到首个 token
    # - 禁用 overlap 后，process(N-1) 在 forward(N) 之前完成
    # - 这样 batch N-1 的首个 token 更早返回给用户
    # ===================================================================
    disable_overlap_for_batch = self._is_disable_overlap_for_batch(batch)

    if disable_overlap_for_batch and self.overlap_executor.has_pending():
      # 连续 prefill 或其他禁用场景：立即处理上一批结果
      with stage("stage::LLMEngine.step_overlap.process_consecutive_prefill"):
        try:
          step_out = self.overlap_executor.process_pending_batch(self._process_overlap_result)
        except Exception as exc:
          log_inference_crash(
            exc, scheduler=self.scheduler, model_runner=self.model_runner,
            stage="step_overlap.process_consecutive_prefill",
          )
          raise
        if step_out:
          if pending_step_out is None:
            pending_step_out = step_out
          else:
            # 合并结果（不应该发生，但防御性处理）
            pending_step_out.outputs.extend(step_out.outputs)

    # ===================================================================
    # Phase 4: run_async — 尽快把 forward(N) 交给 GPU
    # ===================================================================
    # IMPORTANT: avoid GPU sync here (e.g. `(batch.input_ids < 0).any()`),
    # otherwise we will block the CPU scheduler on a CUDA reduction and
    # destroy overlap. Let the scheduler mark placeholder usage explicitly.
    use_placeholder = bool(can_use_placeholder and batch is not None and batch.uses_placeholder)

    current_record = None
    current_forward_batch = None

    if batch is not None and len(batch.reqs) > 0:
      # forward_batch_init 也在 schedule_stream 上执行（构建元数据）
      schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
      try:
        if schedule_stream is not None:
          with torch.cuda.stream(schedule_stream):
            with stage("stage::LLMEngine.step_overlap.forward_batch_init"):
              current_forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)
        else:
          with stage("stage::LLMEngine.step_overlap.forward_batch_init"):
            current_forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)

        with stage("stage::LLMEngine.step_overlap.run_forward_async"):
          current_record = self.overlap_executor.run_forward_async(
            batch,
            current_forward_batch,
            self.model_runner,
            use_placeholder=use_placeholder,
          )
      except Exception as exc:
        log_inference_crash(
          exc,
          scheduler=self.scheduler,
          model_runner=self.model_runner,
          forward_batch=current_forward_batch,
          scheduled_batch=batch,
          stage="step_overlap.forward_init/run_forward_async",
        )
        raise

      if not batch.forward_mode.is_decode():
        # Propagate placeholder metadata for the next schedule() call.
        # This is "scheduler-side" CUDA work; keep it on schedule_stream to avoid
        # accidental default-stream serialization with compute.
        schedule_stream = getattr(self.overlap_executor, "schedule_stream", None)
        if schedule_stream is not None:
          with torch.cuda.stream(schedule_stream):
            self._propagate_future_to_running_batch(batch)
        else:
          self._propagate_future_to_running_batch(batch)

    # ===================================================================
    # Phase 5: Placeholder 路径 — GPU 启动后处理 N-1
    #
    # 此时 GPU 正在执行 forward(N)+sample(N)（如果无 vocab_mask）
    # 或 forward(N)（如果有 vocab_mask）。
    # CPU 并行处理 N-1 的结果，实现真正的 overlap。
    #
    # 注意：如果 disable_overlap_for_batch=True，Phase 3.5 已经处理过，这里跳过
    # ===================================================================
    if (
      can_use_placeholder and not disable_overlap_for_batch and self.overlap_executor.has_pending()
    ):
      with stage("stage::LLMEngine.step_overlap.process_overlap"):
        try:
          pending_step_out = self.overlap_executor.process_pending_batch(self._process_overlap_result)
        except Exception as exc:
          log_inference_crash(
            exc, scheduler=self.scheduler, model_runner=self.model_runner,
            stage="step_overlap.process_overlap",
          )
          raise

    # 收集 pending 处理结果
    if pending_step_out is not None:
      outputs.extend(pending_step_out.outputs)
      num_prefill += pending_step_out.num_prefill_tokens
      num_decode += pending_step_out.num_decode_tokens
      prefill_prefix_lens.update(pending_step_out.prefill_prefix_lens)
      prefill_extend_lens.update(pending_step_out.prefill_extend_lens)

    # ===================================================================
    # Phase 4.5: vocab mask generate (CPU-heavy) + launch sample (GPU-heavy)
    #
    # Only executed when vocab_mask_fn is configured (grammar-constrained sampling).
    # This stage is intentionally after processing (N-1) so any per-request
    # grammar state updates can be reflected in the sampling mask for batch N.
    # ===================================================================
    if batch is not None and len(batch.reqs) > 0 and current_record is not None:
      vocab_mask = None
      with stage("stage::LLMEngine.step_overlap.vocab_mask_generate"):
        if current_forward_batch is not None:
          vocab_mask = self._build_vocab_mask(current_forward_batch)
          if vocab_mask is not None:
            # Transfer the mask on schedule_stream so forward_stream can wait on it.
            if getattr(self.overlap_executor, "schedule_stream", None) is not None:
              with torch.cuda.stream(self.overlap_executor.schedule_stream):
                vocab_mask = vocab_mask.to(self.model_runner.device, non_blocking=True)
            else:
              vocab_mask = vocab_mask.to(self.model_runner.device, non_blocking=True)

      with stage("stage::LLMEngine.step_overlap.launch_sample"):
        try:
          current_record = self.overlap_executor.run_sample_async(
            record=current_record,
            model_runner=self.model_runner,
            vocab_mask=vocab_mask,
          )
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            forward_batch=current_forward_batch,
            scheduled_batch=batch,
            stage="step_overlap.run_sample_async",
          )
          raise

    # Non-overlap mode: process immediately
    if batch is not None and current_record is not None:
      if not getattr(self.overlap_executor, "enable_overlap", True):
        try:
          step_out = self._process_overlap_result(current_record.batch, current_record.batch_result)
        except Exception as exc:
          log_inference_crash(
            exc,
            scheduler=self.scheduler,
            model_runner=self.model_runner,
            forward_batch=current_forward_batch,
            scheduled_batch=current_record.batch,
            stage="step_overlap.process_immediate",
          )
          raise
        if step_out:
          outputs.extend(step_out.outputs)
          num_prefill += step_out.num_prefill_tokens
          num_decode += step_out.num_decode_tokens
          prefill_prefix_lens.update(step_out.prefill_prefix_lens)
          prefill_extend_lens.update(step_out.prefill_extend_lens)

    # 确保 overlap 被禁用或无 batch 时不堆积 pending
    if (not self.config.enable_overlap or batch is None) and self.overlap_executor.has_pending():
      with stage("stage::LLMEngine.step_overlap.sync_pending"):
        while self.overlap_executor.has_pending():
          try:
            step_out = self.overlap_executor.process_pending_batch(self._process_overlap_result)
          except Exception as exc:
            log_inference_crash(
              exc,
              scheduler=self.scheduler,
              model_runner=self.model_runner,
              scheduled_batch=batch,
              stage="step_overlap.sync_pending",
            )
            raise
          if step_out:
            outputs.extend(step_out.outputs)
            num_prefill += step_out.num_prefill_tokens
            num_decode += step_out.num_decode_tokens
            prefill_prefix_lens.update(step_out.prefill_prefix_lens)
            prefill_extend_lens.update(step_out.prefill_extend_lens)

    # Drain deferred releases only after the current batch has already launched
    # its sample/copy work. Cache state must be fresh before the next schedule(),
    # but it does not need to block this step's launch path.
    if getattr(self.scheduler, "pending_release_reqs", None):
      with stage("stage::LLMEngine.step_overlap.drain_pending_releases"):
        self.scheduler.drain_pending_releases()

    self._last_batch = batch

    return StepOutput(
      outputs=outputs,
      num_prefill_tokens=num_prefill,
      num_decode_tokens=num_decode,
      prefill_prefix_lens=prefill_prefix_lens,
      prefill_extend_lens=prefill_extend_lens,
    )

  def generate(
    self,
    prompts: Union[List[str], List[List[int]]],
    sampling_params: Union[SamplingParams, List[SamplingParams]] = None,
    use_tqdm: bool = True,
  ) -> List[Dict[str, Any]]:
    """
    批量生成文本（阻塞式）

    等待所有请求完成后返回结果。
    如需流式输出，请使用 stream_generate()。

    Args:
        prompts: 输入 prompts
        sampling_params: 采样参数
        use_tqdm: 是否显示进度条

    Returns:
        生成结果列表，每个元素包含 text 和 token_ids
    """
    if not self._started:
      self.start()

    # 归一化输入
    if isinstance(prompts, str):
      prompts = [prompts]
    elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], int):
      prompts = [prompts]

    if sampling_params is None:
      sampling_params = SamplingParams()
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)

    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True) if use_tqdm else None

    # 添加所有请求（batch tokenize）
    self.add_requests(prompts, sampling_params)

    # 收集结果
    results: Dict[int, Dict[str, Any]] = {}
    request_ids = []
    decode_throughput = 0.0

    while not self.is_finished():
      t = perf_counter()
      step_output = self.step_overlap()
      step_time = max(perf_counter() - t, 1e-9)

      if pbar:
        if step_output.num_decode_tokens > 0:
          decode_throughput = step_output.num_decode_tokens / step_time
        pbar.set_postfix(
          {
            "Decode": f"{int(decode_throughput)}tok/s",
          }
        )

      for output in step_output.outputs:
        if output.request_id not in results:
          results[output.request_id] = {
            "text": "",
            "token_ids": [],
            "request_id": output.request_id,
          }
          request_ids.append(output.request_id)

        results[output.request_id]["text"] += output.delta_text
        results[output.request_id]["token_ids"] = output.output_token_ids

        if output.finished and pbar:
          pbar.update(1)

    # Flush any remaining pending batches from overlap executor
    while self.overlap_executor.has_pending():
      try:
        step_output = self.overlap_executor.process_pending_batch(self._process_overlap_result)
      except Exception as exc:
        log_inference_crash(
          exc, scheduler=self.scheduler, model_runner=self.model_runner, stage="generate.flush_pending"
        )
        raise
      if step_output:
        for output in step_output.outputs:
          if output.request_id not in results:
            results[output.request_id] = {
              "text": "",
              "token_ids": [],
              "request_id": output.request_id,
            }
            request_ids.append(output.request_id)
          results[output.request_id]["text"] += output.delta_text
          results[output.request_id]["token_ids"] = output.output_token_ids
          if output.finished and pbar:
            pbar.update(1)

    if pbar:
      pbar.close()

    # 按请求 ID 顺序返回结果
    return [results[rid] for rid in sorted(request_ids)]

  # ========== process helper function ============
  def _process_overlap_result(self, batch: ScheduledBatch, result: BatchResult) -> StepOutput:
    """
    处理 overlap 执行中的单个 batch 结果

    作为 OverlapExecutor.process_pending_batch 的回调函数。
    """
    # 处理结果并增量解码
    outputs = self._process_step_result(batch, result)

    # 统计 token 数
    num_prefill = 0
    num_decode = 0
    prefill_prefix_lens: Dict[int, int] = {}
    prefill_extend_lens: Dict[int, int] = {}

    decode_req_set = set()
    if batch.decoding_reqs:
      decode_req_set = {id(r) for r in batch.decoding_reqs}

    if batch.forward_mode.is_extend():
      if batch.prefix_lens and batch.extend_lens:
        for req, pre_len, ext_len in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
          if id(req) in decode_req_set:
            num_decode += 1
          else:
            num_prefill += ext_len
            prefill_prefix_lens[int(req.req_id)] = int(pre_len)
            prefill_extend_lens[int(req.req_id)] = int(ext_len)
      else:
        num_prefill = sum(batch.extend_lens) if batch.extend_lens else 0
    elif batch.forward_mode.is_decode():
      num_decode = len(batch.reqs)

    return StepOutput(
      outputs=outputs,
      num_prefill_tokens=num_prefill,
      num_decode_tokens=num_decode,
      prefill_prefix_lens=prefill_prefix_lens,
      prefill_extend_lens=prefill_extend_lens,
    )

  def _propagate_future_to_running_batch(self, batch: ScheduledBatch) -> None:
    """
    将非 DECODE batch 上的 output_ids (future placeholder) 传播回 running_batch。

    DECODE batch 就是 running_batch 本身，run_batch_async 直接写入 output_ids。
    但 EXTEND / MIXED batch 是新建对象，需要把 batch.output_ids 中 decode 请求对应的 future indices
    以及新加入 running_batch 的 extend 请求的 future indices 传播回 running_batch.output_ids。
    这样下一轮 schedule(N+1) 可以从 running_batch.output_ids 获取 placeholder。

    对于纯 EXTEND batch（chunked_prefill=False），batch 仅包含新 prefill 请求。
    此时 running_batch 中的旧 decode 请求不在 batch 中，但它们在 running_batch.output_ids
    中的旧 future indices 仍然有效（FutureMap 循环 buffer 空间足够），需要保留。
    """
    running_batch = self.scheduler.running_batch
    if batch.output_ids is None or len(running_batch.reqs) == 0:
      running_batch.output_ids = None
      return

    # Save old output_ids before overwriting.
    # _get_new_batch_prefill appends new reqs to running_batch, so:
    #   running_batch.reqs = [old_surviving_reqs..., new_prefill_reqs...]
    #   old_output_ids corresponds to old_surviving_reqs (same order, post-_filter_batch)
    old_output_ids = running_batch.output_ids

    # Build a CPU-side map from running_batch order -> batch indices.
    # Avoid Python for-loop driving per-element GPU writes; do one masked gather on GPU.
    batch_req_map = {id(r): i for i, r in enumerate(batch.reqs)}
    idx_in_batch = [batch_req_map.get(id(req), -1) for req in running_batch.reqs]

    # Validate any "not-in-batch" entries can be sourced from old_output_ids.
    if old_output_ids is None:
      if any(i < 0 for i in idx_in_batch):
        running_batch.output_ids = None
        return
    else:
      old_len = int(old_output_ids.numel())
      for i, idx in enumerate(idx_in_batch):
        if idx < 0 and i >= old_len:
          running_batch.output_ids = None
          return

    device = batch.output_ids.device
    idx_cpu = torch.as_tensor(idx_in_batch, dtype=torch.int64)
    idx_dev = idx_cpu.to(device, non_blocking=True)
    in_batch = idx_dev >= 0

    new_output_ids = torch.empty(
      (len(running_batch.reqs),),
      dtype=batch.output_ids.dtype,
      device=device,
    )

    pos_in_batch = in_batch.nonzero(as_tuple=True)[0]
    if pos_in_batch.numel() > 0:
      src = idx_dev[pos_in_batch]
      new_output_ids[pos_in_batch] = batch.output_ids[src]

    pos_old = (~in_batch).nonzero(as_tuple=True)[0]
    if pos_old.numel() > 0:
      if old_output_ids is None:
        running_batch.output_ids = None
        return
      new_output_ids[pos_old] = old_output_ids[pos_old]

    running_batch.output_ids = new_output_ids

  def _process_step_result(self, batch: ScheduledBatch, result: BatchResult) -> List[RequestOutput]:
    """
    处理单步推理结果，返回每个请求的增量输出

    对于 chunked prefill 请求：
    - 将已推理的 token 插入 radix cache（通过 update_unfinished_req_radix_cache）
    - 不进行 output_ids 解码和 finished 判断
    - 保持 is_chunked 标志直到完成所有 chunk
    """
    if batch is None or len(batch.reqs) == 0:
      return []

    next_token_ids = result.next_token_ids
    if isinstance(next_token_ids, torch.Tensor):
      # In overlap mode, next_token_ids is already on CPU (cloned from pinned buffer).
      # Check device to avoid unnecessary .cpu() call which could disguise issues.
      if next_token_ids.device.type != "cpu":
        logger.error(
          "Expected next_token_ids to be on CPU, but got device: %s", next_token_ids.device
        )
        next_token_ids = next_token_ids.cpu()
      next_token_ids = next_token_ids.tolist()

    outputs = []
    finished_req_ids = []

    # Decode tokens in batch to reduce Python overhead.
    active_reqs: List[Req] = []
    active_req_ids: List[int] = []
    active_token_ids: List[int] = []

    for i, req in enumerate(batch.reqs):
      # ============ Chunked Prefill 特殊处理 ============
      # 对于未完成 prefill 的 chunked 请求，只更新 radix cache，
      # 不进行 output_ids 解码和 finished 判断
      if req.is_chunked:
        # 将已推理的 token 插入 radix cache
        # update_unfinished_req_radix_cache 会：
        # 1. 将当前 fill_ids 对应的 KV 写入 radix cache
        # 2. 恢复 fill_ids 为完整序列（origin_input_ids + output_ids），以便后续 chunk 正确切片
        # 3. 释放之前的 req_pool_idx，已经写入了 radix cache
        self.kv_cache_mgr.update_unfinished_req_radix_cache(req)
        req.fill_ids = req.origin_input_ids + req.output_ids
        self.kv_cache_mgr.request_pool.free([req.req_pool_idx])
        req.req_pool_idx = -1
        logger.debug(
          f"Updated radix cache for chunked prefill req_id={req.req_id}, "
          f"cache_protected_len={req.cache_protected_len}, "
          f"prefix_indices={req.prefix_indices}, "
          f"last_node={req.last_node}, "
          f"req_pool_idx=" + str(req.req_pool_idx)
          if req.req_pool_idx != -1
          else "no req_pool_idx"
        )
        continue

      # ============ 正常请求处理 ============
      # overlap placeholder 路径下可能出现 "stale batch"：
      # schedule(N) 先于 process(N-1)，导致已经在 (N-1) 结束的请求仍出现在 batch(N) 的 record 中。
      # 这时必须跳过该请求，避免 output_ids 超过 max_tokens 或重复解码。
      if req.finished:
        continue

      # BUG Fix: retract 后请求的 KV cache 已被释放，output_ids 已重置。
      # schedule(N) 的 retract 先于 process(N-1)，此时不应追加 token
      # 到已重置的 output_ids，否则下次 prefill 会包含孤立 token 导致乱码。
      if req.is_retracted:
        self.detokenizer.cleanup(req.req_id)
        req.is_retracted = False
        continue

      token_id = next_token_ids[i]
      req.append_output_token(token_id)

      active_reqs.append(req)
      active_req_ids.append(req.req_id)
      active_token_ids.append(token_id)

    if active_reqs:
      decoded = self.detokenizer.decode_batch(
        req_ids=active_req_ids,
        token_ids=active_token_ids,
        eos_token_id=self.scheduler.eos_token_id,
        return_full_text=True,
      )
      for req, token_id, (delta_text, is_eos, full_text) in zip(
        active_reqs, active_token_ids, decoded
      ):
        # 检查是否达到最大 token 数
        if req.ignore_eos:
          is_eos = False
        is_finished = is_eos or len(req.output_ids) >= req.max_tokens
        finish_reason = None

        if is_finished:
          # flush() is typically a no-op after decode(); keep for safety.
          delta_text += self.detokenizer.flush(req.req_id)

          req.finished = True
          if is_eos:
            finish_reason = "eos"
            req.finished_reason = "eos"
          else:
            finish_reason = "max_tokens"
            req.finished_reason = "max_tokens"
          finished_req_ids.append(req.req_id)

        outputs.append(
          RequestOutput(
            request_id=req.req_id,
            delta_text=delta_text,
            full_text=full_text,
            token_id=token_id,
            output_token_ids=req.output_ids,
            finished=is_finished,
            finish_reason=finish_reason,
          )
        )
        if is_finished:
          # Drop detokenizer state to reduce per-step CPU/memory overhead.
          self.detokenizer.cleanup(req.req_id)

    # 处理完成的请求
    self.scheduler._handle_finished_requests(batch, finished_req_ids)

    return outputs

  def is_finished(self) -> bool:
    """检查是否所有请求都已完成"""
    return not self.scheduler.has_unfinished()

  # =========== start and stop ========
  def start(self):
    """启动引擎"""
    if self._started:
      return

    self._started = True
    logger.info("Engine started")

  def stop(self):
    """停止引擎"""
    if not self._started:
      return

    logger.info("Stopping engine...")
    # 单进程模式
    if hasattr(self, "model_runner"):
      # ModelRunner currently has no IPC; just drop the reference
      del self.model_runner

    for p in self.ps:
      p.join()

    self._started = False
    logger.info("Engine stopped")

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()

  # ================= for test =================
  def add_request(
    self,
    prompt: Union[str, List[int]],
    sampling_params: Optional[SamplingParams] = None,
  ) -> int:
    """
    添加请求到调度队列（非阻塞）

    新请求会被添加到等待队列，不会阻塞当前正在进行的推理。
    调用 step() 或 stream_generate() 来驱动推理。

    Args:
        prompt: 输入 prompt (文本或 token ids)
        sampling_params: 采样参数

    Returns:
        request_id: 请求 ID，用于后续查询状态和结果
    """
    if sampling_params is None:
      sampling_params = SamplingParams()

    if isinstance(prompt, str):
      with stage("stage::Tokenizer.encode_one"):
        token_ids = self.prompt_tokenizer.encode_one(prompt)
    else:
      token_ids = prompt

    req = Req(token_ids, sampling_params)
    self.scheduler.add(req)
    self._req_by_id[req.req_id] = req
    return req.req_id

  def stream_generate(
    self,
    prompts: Union[str, List[str], List[int], List[List[int]]],
    sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]] = None,
  ) -> Generator[RequestOutput, None, None]:
    """
    流式生成接口（生成器）

    每次 yield 一个请求的增量输出。可以在生成过程中通过
    add_request() 添加新请求，新请求会被自动纳入调度。

    Args:
        prompts: 输入 prompts（支持单个或列表，文本或 token ids）
        sampling_params: 采样参数

    Yields:
        RequestOutput: 每个请求的增量输出

    Example:
        >>> for output in engine.stream_generate(["Hello", "World"]):
        ...   print(f"[{output.request_id}] {output.delta_text}", end="")
        ...   if output.finished:
        ...     print()
    """
    if not self._started:
      self.start()

    # 归一化输入`
    if isinstance(prompts, str):
      prompts = [prompts]
    elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], int):
      prompts = [prompts]  # 单个 token ids 列表

    if sampling_params is None:
      sampling_params = SamplingParams()
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)

    # 添加所有初始请求（batch tokenize）
    self.add_requests(prompts, sampling_params)

    # 持续推理直到所有请求完成
    while self.scheduler.has_unfinished():
      step_output = self.step_overlap()

      for output in step_output.outputs:
        yield output

    # Flush any remaining pending batches from overlap executor
    while self.overlap_executor.has_pending():
      try:
        step_output = self.overlap_executor.process_pending_batch(self._process_overlap_result)
      except Exception as exc:
        log_inference_crash(
          exc, scheduler=self.scheduler, model_runner=self.model_runner, stage="stream_generate.flush_pending"
        )
        raise
      if step_output:
        for output in step_output.outputs:
          yield output
