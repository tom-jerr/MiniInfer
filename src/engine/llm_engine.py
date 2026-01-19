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
from dataclasses import fields
from time import perf_counter
from typing import List, Optional, Union, Dict, Any
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import zmq

from .engine_config import Config
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .seqeunce import Sequence
from .process_controller import ProcessController
from .ipc.zmq_channel import create_socket, send_pyobj, recv_pyobj
from .ipc.protocol import Request, Response, MessageType, GenerateRequest
from ..utils.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


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
        # 提取出 Config 类真正定义了的参数
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        self.config = Config(model, **config_kwargs)

        self.use_multiprocess = use_multiprocess
        self._started = False

        if use_multiprocess:
            self._init_multiprocess_mode(**kwargs)
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

        for i in range(1, self.config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(self.config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # Model Runner (rank 0)
        self.model_runner = ModelRunner(self.config, 0, self.events)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
            trust_remote_code=True,
        )
        self.config.eos = self.tokenizer.eos_token_id

        # Scheduler
        self.scheduler = Scheduler(self.config)

        self._started = True

    def _init_multiprocess_mode(self, **kwargs):
        """初始化多进程模式"""
        logger.info("Initializing LLMEngine in multi-process mode")

        # 提取多进程相关配置
        mp_config = {
            "model_path": self.config.model,
            "max_tokens": kwargs.get("max_tokens", 65536),
            "max_requests": kwargs.get("max_requests", 256),
            "max_context_len": kwargs.get("max_context_len", 4096),
            "num_layers": kwargs.get("num_layers", 32),
            "num_heads": kwargs.get("num_heads", 32),
            "head_dim": kwargs.get("head_dim", 128),
            "max_batch_size": kwargs.get("max_batch_size", 32),
            "max_prefill_tokens": kwargs.get("max_prefill_tokens", 4096),
            "enable_prefix_cache": kwargs.get("enable_prefix_cache", True),
        }

        # 创建进程控制器
        self.process_controller = ProcessController(**mp_config)

        # ZMQ 客户端 socket
        self.zmq_context: Optional[zmq.Context] = None
        self.client_socket: Optional[zmq.Socket] = None
        self.response_socket: Optional[zmq.Socket] = None

        # 待处理的请求
        self.pending_requests: Dict[str, Any] = {}

    def start(self):
        """启动引擎 (多进程模式)"""
        if self._started:
            return

        if self.use_multiprocess:
            logger.info("Starting multi-process engine...")

            # 启动所有 Worker 进程
            self.process_controller.start_all()

            # 等待进程初始化
            import time

            time.sleep(1.0)

            # 初始化客户端 socket
            self.zmq_context = zmq.Context()

            # 连接到 Tokenizer (发送请求)
            self.client_socket = create_socket(
                self.zmq_context,
                zmq.DEALER,
                self.process_controller.ipc_addresses["client_to_tokenizer"],
                bind=False,
                identity=f"client-{uuid.uuid4().hex[:8]}",
            )

            # 连接到 Detokenizer (接收响应)
            self.response_socket = create_socket(
                self.zmq_context,
                zmq.DEALER,
                self.process_controller.ipc_addresses["detokenizer_to_client"],
                bind=False,
                identity=f"client-{uuid.uuid4().hex[:8]}",
            )

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
        sampling_params: SamplingParams,
    ) -> str:
        """
        添加请求

        Args:
            prompt: 输入 prompt (文本或 token ids)
            sampling_params: 采样参数

        Returns:
            request_id
        """
        request_id = uuid.uuid4().hex

        if self.use_multiprocess:
            # 多进程模式：发送给 Tokenizer
            gen_request = GenerateRequest(
                request_id=request_id,
                prompt=(
                    prompt if isinstance(prompt, str) else self.tokenizer.decode(prompt)
                ),
                sampling_params=sampling_params,
            )

            request = Request(
                request_id=request_id,
                msg_type=MessageType.GENERATE_REQUEST,
                payload=gen_request,
            )

            send_pyobj(self.client_socket, request)
            self.pending_requests[request_id] = {"status": "pending"}
        else:
            # 单进程模式
            if isinstance(prompt, str):
                token_ids = self.tokenizer.encode(prompt)
            else:
                token_ids = prompt

            seq = Sequence(token_ids, sampling_params)
            self.scheduler.add(seq)

        return request_id

    def step(self):
        """
        执行一步推理

        Returns:
            output: List[(seq_id, token_ids)] - 已完成的序列
            num_tokens: int - 本次处理的 token 数，正数表示 prefill，负数表示 decode
        """
        if self.use_multiprocess:
            # 多进程模式：从 response socket 接收结果
            # 实际推理由 Scheduler Worker 驱动
            return [], 0
        else:
            # 单进程模式
            # 1. 调度获取 batch
            batch = self.scheduler.schedule()
            if batch is None or batch.is_empty:
                return [], 0

            # 2. 执行 forward
            logits = self.model_runner.forward(batch)

            # 3. 采样
            next_tokens = self.model_runner.sample(logits, batch)

            # 4. 更新状态
            finished_seqs = self.scheduler.update_after_step(batch, next_tokens)

            # 5. 返回结果
            output = []
            for seq in finished_seqs:
                output.append((seq.seq_id, seq.get_output_token_ids()))

            # 计算 token 数: prefill 为正，decode 为负
            from .scheduler import BatchType

            if batch.batch_type == BatchType.PREFILL_ONLY:
                num_tokens = batch.num_prefill_tokens
            elif batch.batch_type == BatchType.DECODE_ONLY:
                num_tokens = -batch.num_decode_tokens
            else:  # MIXED
                num_tokens = batch.num_prefill_tokens  # 以 prefill 为主

            return output, num_tokens

    def is_finished(self) -> bool:
        """检查是否所有请求都已完成"""
        if self.use_multiprocess:
            return len(self.pending_requests) == 0
        else:
            return not self.scheduler.has_unfinished()

    def generate(
        self,
        prompts: Union[List[str], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        生成文本

        Args:
            prompts: 输入 prompts
            sampling_params: 采样参数
            use_tqdm: 是否显示进度条

        Returns:
            生成结果列表
        """
        if not self._started:
            self.start()

        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # 添加所有请求
        request_ids = []
        for prompt, param in zip(prompts, sampling_params):
            request_id = self.add_request(prompt, param)
            request_ids.append(request_id)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        # 等待所有请求完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            if use_tqdm:
                if num_tokens and num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                elif num_tokens:
                    decode_throughput = -num_tokens / (perf_counter() - t)

                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )

            if output:
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)

        if use_tqdm:
            pbar.close()

        # 格式化输出
        outputs = [outputs.get(seq_id, []) for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]

        return outputs

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
