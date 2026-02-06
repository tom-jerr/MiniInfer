"""
LLMEngine - 对外暴露的统一接口

两种运行模式:
1. 单进程模式 (Simple): 所有组件在同一进程内运行
2. 多进程模式 (Distributed): Tokenizer/Scheduler/Detokenizer 分别在独立进程

Usage:
    # 单进程模式 (默认)
    engine = LLMEngine(model_path="...")
    outputs = engine.generate(prompts, sampling_params)

    # 多进程模式
    engine = LLMEngine(model_path="...", use_multiprocess=True)
    engine.start()
    outputs = engine.generate(prompts, sampling_params)
    engine.stop()
"""

import atexit
import logging
import uuid
from dataclasses import dataclass, field, fields
from time import perf_counter
from typing import List, Optional, Union, Dict, Any, Iterator, Generator
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
import zmq

from miniinfer.kvcache.kv_cache_manager import KVCacheManager
from miniinfer.config.engine.config import EngineConfig
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .scheduler_batch import Req, ScheduledBatch, ForwardBatch, BatchResult

# from .process_controller import ProcessController
# from .ipc.zmq_channel import create_socket, send_pyobj, recv_pyobj
# from .ipc.protocol import Request, Response, MessageType, GenerateRequest
from ..utils.sampling_params import SamplingParams
from .detokenizer import IncrementalDecoder

logger = logging.getLogger(__name__)


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


class LLMEngine:
    """
    LLM 推理引擎

    支持两种模式:
    - 单进程模式: 所有组件在同一进程，适合调试和小规模部署
    - 多进程模式: Tokenizer/Scheduler/Detokenizer 分离，适合生产环境
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

        self.use_multiprocess = use_multiprocess
        self._started = False

        if use_multiprocess:
            pass
        else:
            self._init_single_process_mode()

        atexit.register(self.stop)

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
                extend_lens = forward_batch.extend_seq_lens_cpu or []
                last_token_indices: List[int] = []
                cumsum = 0
                for length in extend_lens:
                    last_token_indices.append(cumsum + int(length) - 1)
                    cumsum += int(length)
                if not last_token_indices:
                    return
                idx = torch.tensor(last_token_indices, device=logits.device)
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
                print(
                    "req=%s top%d next_token candidates: %s", req.req_id, k, candidates
                )

    def _should_apply_chat_template(self) -> bool:
        if self.use_chat_template is True:
            return hasattr(self.tokenizer, "apply_chat_template")
        if self.use_chat_template is False:
            return False
        # auto
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return False
        model_name = str(getattr(self.config, "model", "")).lower()
        return ("instruct" in model_name) or ("chat" in model_name)

    def _encode_prompt(self, prompt: str) -> List[int]:
        """
        Encode a user prompt into token ids.

        For chat/instruct models, use the tokenizer chat template (when available)
        so the model sees the expected system/user/assistant framing.
        """
        if self._should_apply_chat_template():
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(encoded, torch.Tensor):
                    return encoded.tolist()
                if isinstance(encoded, dict) and "input_ids" in encoded:
                    ids = encoded["input_ids"]
                    if isinstance(ids, torch.Tensor):
                        return ids.tolist()
                    return list(ids)
                return list(encoded)
            except Exception as e:
                logger.warning(
                    "apply_chat_template failed (%s); falling back to tokenizer.encode",
                    e,
                )
        return self.tokenizer.encode(prompt)

    def _init_single_process_mode(self):
        """初始化单进程模式"""
        logger.info("Initializing LLMEngine in single-process mode")

        # TP 子进程
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        # for i in range(1, self.config.tensor_parallel_size):
        #     event = ctx.Event()
        #     process = ctx.Process(target=ModelRunner, args=(self.config, i, event))
        #     process.start()
        #     self.ps.append(process)
        #     self.events.append(event)
        config = self.config

        # 初始化 KV Cache 管理器
        # 使用 MemoryBudgetManager 自动计算 max_total_tokens
        self.kv_cache_mgr = KVCacheManager(
            size=config.max_total_tokens,  # None 时自动根据 GPU 显存计算
            max_requests=config.max_num_seqs,
            max_context_len=config.max_context_len,
            num_layers=config.hf_config.num_hidden_layers,
            num_heads=config.hf_config.num_key_value_heads,  # 使用 KV head 数量，支持 GQA
            head_dim=config.hf_config.hidden_size
            // config.hf_config.num_attention_heads,
            dtype=config.dtype,
            device="cuda",
            enable_prefix_cache=config.enable_prefix_cache,
            page_size=config.page_size,
            max_extend_tokens=config.max_extend_len,
            # 显存预算相关配置
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
        )

        # 将实际计算的 max_total_tokens 回写到 config
        config.max_total_tokens = self.kv_cache_mgr.size

        # Model Runner (rank 0)
        self.model_runner = ModelRunner(self.config, self.kv_cache_mgr, 0, self.events)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
            trust_remote_code=True,
        )
        # Stream detokenizer
        self.detokenizer = IncrementalDecoder(
            tokenizer=self.tokenizer,
            skip_special_tokens=True,
        )
        self.config.eos = self.tokenizer.eos_token_id

        # Scheduler (传入 tokenizer 和 model_runner)
        self.scheduler = Scheduler(self.config, self.tokenizer, self.kv_cache_mgr)

        self._started = True

        # 打印内存预算信息
        budget_stats = self.kv_cache_mgr.memory_budget.get_stats()
        logger.info(
            f"Memory budget initialized: "
            f"max_total_tokens={budget_stats['max_total_tokens']}, "
            f"num_pages={budget_stats['num_pages']}, "
            f"kv_cache_memory={budget_stats['kv_cache_memory_gb']:.2f}GB, "
            f"max_num_batched_tokens={budget_stats['max_num_batched_tokens']}, "
            f"gpu_memory_utilization={budget_stats['gpu_memory_utilization']}"
        )

    def start(self):
        """启动引擎 (多进程模式)"""
        if self._started:
            return

        self._started = True
        logger.info("Engine started")

    def stop(self):
        """停止引擎"""
        if not self._started:
            return

        logger.info("Stopping engine...")

        if self.use_multiprocess:
            # 关闭客户端 socket
            if self.client_socket:
                self.client_socket.close()
            if self.response_socket:
                self.response_socket.close()
            if self.zmq_context:
                self.zmq_context.term()

            # 停止所有 Worker 进程
            self.process_controller.stop_all()
        else:
            # 单进程模式
            if hasattr(self, "model_runner"):
                # ModelRunner currently has no IPC; just drop the reference
                del self.model_runner

            for p in self.ps:
                p.join()

        self._started = False
        logger.info("Engine stopped")

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
            token_ids = self._encode_prompt(prompt)
        else:
            token_ids = prompt

        req = Req(token_ids, sampling_params)
        self.scheduler.add(req)
        return req.req_id

    def step(self) -> StepOutput:
        """
        执行单步推理（非阻塞式）

        每次调用执行一个推理步骤：调度 -> forward -> sample -> 增量解码。
        新请求可以在任意时刻通过 add_request() 添加，不会阻塞此方法。

        Returns:
            StepOutput: 包含本步所有请求的增量输出
        """
        if self.use_multiprocess:
            # 多进程模式：从 response socket 接收结果
            return StepOutput(outputs=[])

        # 单进程模式
        # 1. 调度获取 batch
        batch = self.scheduler.schedule(self.model_runner.device)
        if batch is None or len(batch.reqs) == 0:
            return StepOutput(outputs=[])

        # 2. 执行 forward
        forward_batch = ForwardBatch.init_new(batch, self.model_runner.attn_backend)
        output = self.model_runner.forward(forward_batch)
        logits = output.logits
        # if self.debug_logit_topk > 0:
        # self._debug_print_topk_next_tokens(logits, forward_batch)
        # 3. 采样
        next_tokens = self.model_runner.sample(logits, forward_batch)

        # 4. 处理结果并增量解码
        result = BatchResult(logits=logits, next_token_ids=next_tokens)
        outputs = self._process_step_result(batch, result)

        # 5. 统计 token 数
        num_prefill = 0
        num_decode = 0
        prefill_prefix_lens: Dict[int, int] = {}
        prefill_extend_lens: Dict[int, int] = {}
        if batch.forward_mode.is_extend():
            num_prefill = sum(batch.extend_lens) if batch.extend_lens else 0
            if batch.prefix_lens and batch.extend_lens:
                for req, pre_len, ext_len in zip(
                    batch.reqs, batch.prefix_lens, batch.extend_lens
                ):
                    prefill_prefix_lens[int(req.req_id)] = int(pre_len)
                    prefill_extend_lens[int(req.req_id)] = int(ext_len)
        elif batch.forward_mode.is_decode():
            num_decode = len(batch.reqs)

        return StepOutput(
            outputs=outputs,
            num_prefill_tokens=num_prefill,
            num_decode_tokens=num_decode,
            prefill_prefix_lens=prefill_prefix_lens,
            prefill_extend_lens=prefill_extend_lens,
        )

    def _process_step_result(
        self, batch: ScheduledBatch, result: BatchResult
    ) -> List[RequestOutput]:
        """
        处理单步推理结果，返回每个请求的增量输出
        """
        if batch is None or len(batch.reqs) == 0:
            return []

        next_token_ids = result.next_token_ids
        if isinstance(next_token_ids, torch.Tensor):
            next_token_ids = next_token_ids.cpu().tolist()

        outputs = []
        finished_req_ids = []

        for i, req in enumerate(batch.reqs):
            token_id = next_token_ids[i]
            req.output_ids.append(token_id)

            # 增量解码
            delta_text, is_eos = self.detokenizer.decode(
                req_id=req.req_id,
                token_id=token_id,
                eos_token_id=self.scheduler.eos_token_id,
            )

            # 检查是否达到最大 token 数
            if req.ignore_eos:
                is_eos = False
            is_finished = is_eos or len(req.output_ids) >= req.max_tokens
            finish_reason = None

            if is_finished:
                # 刷新剩余文本
                remaining = self.detokenizer.flush(req.req_id)
                delta_text += remaining

                req.finished = True
                if is_eos:
                    finish_reason = "eos"
                    req.finished_reason = "eos"
                else:
                    finish_reason = "max_tokens"
                    req.finished_reason = "max_tokens"
                finished_req_ids.append(req.req_id)

            # 获取完整文本
            full_text = self.detokenizer.get_full_text(req.req_id)

            outputs.append(
                RequestOutput(
                    request_id=req.req_id,
                    delta_text=delta_text,
                    full_text=full_text,
                    token_id=token_id,
                    output_token_ids=req.output_ids.copy(),
                    finished=is_finished,
                    finish_reason=finish_reason,
                )
            )

        # 处理完成的请求
        self.scheduler._handle_finished_requests(batch, finished_req_ids)

        return outputs

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
            ...     print(f"[{output.request_id}] {output.delta_text}", end="")
            ...     if output.finished:
            ...         print()
        """
        if not self._started:
            self.start()

        # 归一化输入
        if isinstance(prompts, str):
            prompts = [prompts]
        elif (
            isinstance(prompts, list)
            and len(prompts) > 0
            and isinstance(prompts[0], int)
        ):
            prompts = [prompts]  # 单个 token ids 列表

        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有初始请求
        for prompt, param in zip(prompts, sampling_params):
            self.add_request(prompt, param)

        # 持续推理直到所有请求完成
        while self.scheduler.has_unfinished():
            step_output = self.step()

            for output in step_output.outputs:
                yield output

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

        if sampling_params is None:
            sampling_params = SamplingParams()

        pbar = None
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # 收集结果
        results: Dict[int, Dict[str, Any]] = {}
        request_ids = []

        prefill_throughput = decode_throughput = 0.0

        # 使用流式生成收集结果
        for output in self.stream_generate(prompts, sampling_params):
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

    def get_request_status(self, request_id: int) -> Dict[str, Any]:
        """
        获取请求状态

        Args:
            request_id: 请求 ID

        Returns:
            请求状态字典，包含 status、output_tokens、finished 等
        """
        if request_id not in self._request_map:
            return {"status": "not_found", "request_id": request_id}

        req = self._request_map[request_id]

        if req.finished:
            status = "finished"
        elif req in self.scheduler.running_batch.reqs:
            status = "running"
        elif req in self.scheduler.waiting_queue:
            status = "waiting"
        else:
            status = "unknown"

        return {
            "status": status,
            "request_id": request_id,
            "output_token_count": len(req.output_ids),
            "finished": req.finished,
            "finish_reason": req.finished_reason if req.finished else None,
        }

    def get_request_output(self, request_id: int) -> Optional[str]:
        """
        获取请求的完整输出文本

        Args:
            request_id: 请求 ID

        Returns:
            完整输出文本，如果请求不存在则返回 None
        """
        return self.scheduler.get_request_output(request_id)

    def get_finished_requests(self) -> List[Req]:
        """
        获取并清空已完成的请求列表

        Returns:
            已完成的请求列表
        """
        finished = self.scheduler.finished_reqs.copy()
        self.scheduler.finished_reqs.clear()
        return finished

    def is_finished(self) -> bool:
        """检查是否所有请求都已完成"""
        if self.use_multiprocess:
            return len(self.pending_requests) == 0
        else:
            return not self.scheduler.has_unfinished()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.use_multiprocess:
            return self.process_controller.get_stats()
        else:
            return {
                "mode": "single-process",
                "started": self._started,
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
