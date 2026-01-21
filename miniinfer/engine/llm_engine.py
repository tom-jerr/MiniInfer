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
from dataclasses import dataclass, fields
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

        self.use_multiprocess = use_multiprocess
        self._started = False

        if use_multiprocess:
           pass 
        else:
            self._init_single_process_mode()

        atexit.register(self.stop)

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
        self.kv_cache_mgr = KVCacheManager(
            size=config.max_total_tokens,
            max_requests=config.max_num_seqs,
            max_context_len=config.max_context_len,
            num_layers=config.hf_config.num_hidden_layers,
            num_heads=config.hf_config.num_attention_heads,
            head_dim=config.hf_config.hidden_size
            // config.hf_config.num_attention_heads,
        )
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
        self.scheduler = Scheduler(
            self.config, 
            self.tokenizer, 
            self.kv_cache_mgr
        )
        
        self._started = True

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
                self.model_runner.call("exit")
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
            token_ids = self.tokenizer.encode(prompt)
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
        model_output = self.model_runner.forward(forward_batch)
        logits = model_output.logits

        # 3. 采样
        next_tokens = self.model_runner.sample(logits, forward_batch)

        # 4. 处理结果并增量解码
        result = BatchResult(logits=logits, next_token_ids=next_tokens)
        outputs = self._process_step_result(batch, result)

        # 5. 统计 token 数
        num_prefill = 0
        num_decode = 0
        if batch.forward_mode.is_extend():
            num_prefill = sum(batch.extend_lens) if batch.extend_lens else 0
        elif batch.forward_mode.is_decode():
            num_decode = len(batch.reqs)
        
        return StepOutput(
            outputs=outputs,
            num_prefill_tokens=num_prefill,
            num_decode_tokens=num_decode,
        )

    def _process_step_result(
        self, 
        batch: ScheduledBatch, 
        result: BatchResult
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
            delta_text, is_eos = self.scheduler.incremental_decoder.decode(
                req_id=req.req_id,
                token_id=token_id,
                eos_token_id=self.scheduler.eos_token_id,
            )

            # 检查是否达到最大 token 数
            is_finished = is_eos or len(req.output_ids) >= req.max_tokens
            finish_reason = None

            if is_finished:
                # 刷新剩余文本
                remaining = self.scheduler.incremental_decoder.flush(req.req_id)
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
            full_text = self.scheduler.incremental_decoder.get_full_text(req.req_id)

            outputs.append(RequestOutput(
                request_id=req.req_id,
                delta_text=delta_text,
                full_text=full_text,
                token_id=token_id,
                output_token_ids=req.output_ids.copy(),
                finished=is_finished,
                finish_reason=finish_reason,
            ))

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
        elif isinstance(prompts, list) and len(prompts) > 0 and isinstance(prompts[0], int):
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
