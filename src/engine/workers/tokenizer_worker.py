"""Tokenizer Worker 进程"""

import logging
from typing import Optional, Dict, Any
from transformers import AutoTokenizer
import zmq

from .base import BaseWorker, WorkerState
from ..ipc.zmq_channel import create_socket, send_pyobj, recv_pyobj
from ..ipc.protocol import (
    MessageType,
    Request,
    Response,
    TokenizeRequest,
    TokenizeResult,
    GenerateRequest,
)

logger = logging.getLogger(__name__)


class TokenizerWorker(BaseWorker):
    """
    Tokenizer 进程

    职责:
    - 接收客户端的生成请求
    - 将 prompt 转换为 token ids
    - 将 tokenize 结果发送给 Scheduler
    """

    def __init__(
        self,
        model_path: str,
        client_address: str,  # 接收客户端请求
        scheduler_address: str,  # 发送给 Scheduler
        zmq_context: Optional[zmq.Context] = None,
    ):
        super().__init__("TokenizerWorker", zmq_context)
        self.model_path = model_path
        self.client_address = client_address
        self.scheduler_address = scheduler_address

        self.tokenizer: Optional[AutoTokenizer] = None
        self.client_socket: Optional[zmq.Socket] = None
        self.scheduler_socket: Optional[zmq.Socket] = None

        # 请求映射: request_id -> client_identity
        self.pending_requests: Dict[str, bytes] = {}

    def setup(self):
        """初始化 tokenizer 和 ZMQ socket"""
        # 加载 tokenizer
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )

        # 设置客户端 socket (ROUTER 用于处理多个客户端)
        self.client_socket = create_socket(
            self.zmq_context,
            zmq.ROUTER,
            self.client_address,
            bind=True,
            identity="tokenizer",
        )

        # 设置 Scheduler socket (PUSH 发送)
        self.scheduler_socket = create_socket(
            self.zmq_context, zmq.PUSH, self.scheduler_address, bind=False
        )

        logger.info("TokenizerWorker setup complete")

    def process(self) -> bool:
        """处理一轮消息"""
        # 从客户端接收请求 (非阻塞)
        result = recv_pyobj(self.client_socket, timeout=100)  # 100ms 超时
        if result is None:
            return True

        identity, data = result

        if isinstance(data, Request):
            self._handle_request(identity, data)

        return True

    def _handle_request(self, identity: bytes, request: Request):
        """处理请求"""
        if request.msg_type == MessageType.GENERATE_REQUEST:
            self._handle_generate_request(identity, request)
        elif request.msg_type == MessageType.HEALTH_CHECK:
            self._handle_health_check(identity, request)
        elif request.msg_type == MessageType.SHUTDOWN:
            self._shutdown_event.set()

    def _handle_generate_request(self, identity: bytes, request: Request):
        """处理生成请求"""
        gen_request: GenerateRequest = request.payload

        # Tokenize
        token_ids = self.tokenizer.encode(gen_request.prompt)

        # 记录请求来源
        self.pending_requests[gen_request.request_id] = identity

        # 发送给 Scheduler
        tokenize_result = TokenizeResult(
            request_id=gen_request.request_id,
            token_ids=token_ids,
            sampling_params=gen_request.sampling_params,
        )

        scheduler_request = Request(
            request_id=gen_request.request_id,
            msg_type=MessageType.TOKENIZE_RESULT,
            payload=tokenize_result,
        )

        send_pyobj(self.scheduler_socket, scheduler_request)
        logger.debug(
            f"Tokenized request {gen_request.request_id}: {len(token_ids)} tokens"
        )

    def _handle_health_check(self, identity: bytes, request: Request):
        """处理健康检查"""
        response = Response(
            request_id=request.request_id,
            msg_type=MessageType.HEALTH_RESPONSE,
            payload={"state": self.state.name},
        )
        send_pyobj(self.client_socket, response, identity=identity)

    def cleanup(self):
        """清理资源"""
        if self.client_socket:
            self.client_socket.close()
        if self.scheduler_socket:
            self.scheduler_socket.close()
