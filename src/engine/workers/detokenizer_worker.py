"""Detokenizer Worker 进程"""

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
    DetokenizeRequest,
    DetokenizeResult,
    GenerateResponse,
)

logger = logging.getLogger(__name__)


class DetokenizerWorker(BaseWorker):
    """
    Detokenizer 进程

    职责:
    - 从 Scheduler 接收生成的 token ids
    - 将 token ids 转换为文本
    - 将结果发送回客户端
    """

    def __init__(
        self,
        model_path: str,
        scheduler_address: str,  # 从 Scheduler 接收
        client_address: str,  # 发送给客户端 (通过 Tokenizer 转发)
        zmq_context: Optional[zmq.Context] = None,
    ):
        super().__init__("DetokenizerWorker", zmq_context)
        self.model_path = model_path
        self.scheduler_address = scheduler_address
        self.client_address = client_address

        self.tokenizer: Optional[AutoTokenizer] = None
        self.scheduler_socket: Optional[zmq.Socket] = None
        self.client_socket: Optional[zmq.Socket] = None

        # 增量 detokenize 状态
        self.decode_states: Dict[str, dict] = {}

    def setup(self):
        """初始化"""
        # 加载 tokenizer
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            use_fast=True,
            trust_remote_code=True,
        )

        # 设置 Scheduler socket (PULL 接收)
        self.scheduler_socket = create_socket(
            self.zmq_context, zmq.PULL, self.scheduler_address, bind=True
        )

        # 设置客户端 socket (ROUTER 用于发送响应)
        self.client_socket = create_socket(
            self.zmq_context,
            zmq.ROUTER,
            self.client_address,
            bind=True,
            identity="detokenizer",
        )

        logger.info("DetokenizerWorker setup complete")

    def process(self) -> bool:
        """处理一轮消息"""
        result = recv_pyobj(self.scheduler_socket, timeout=100)
        if result is None:
            return True

        _, data = result

        if isinstance(data, Request):
            if data.msg_type == MessageType.DETOKENIZE_REQUEST:
                self._handle_detokenize_request(data)
            elif data.msg_type == MessageType.SHUTDOWN:
                self._shutdown_event.set()

        return True

    def _handle_detokenize_request(self, request: Request):
        """处理 detokenize 请求"""
        detok_request: DetokenizeRequest = request.payload

        # Detokenize
        text = self.tokenizer.decode(
            detok_request.token_ids,
            skip_special_tokens=detok_request.skip_special_tokens,
        )

        # 创建响应
        response = GenerateResponse(
            request_id=detok_request.request_id,
            text=text,
            token_ids=detok_request.token_ids,
            finished=True,
        )

        # 发送响应 (这里简化，实际应该路由到正确的客户端)
        resp = Response(
            request_id=detok_request.request_id,
            msg_type=MessageType.GENERATE_RESPONSE,
            payload=response,
        )

        # TODO: 需要知道客户端 identity 才能正确路由
        # 这里可以通过 request_id 查找
        logger.debug(
            f"Detokenized request {detok_request.request_id}: {len(text)} chars"
        )

    def _incremental_detokenize(
        self,
        request_id: str,
        token_ids: list,
    ) -> str:
        """
        增量 detokenize

        用于流式输出场景
        """
        if request_id not in self.decode_states:
            self.decode_states[request_id] = {
                "prev_tokens": [],
                "prev_text": "",
            }

        state = self.decode_states[request_id]

        # 解码所有 tokens
        full_text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

        # 计算增量
        delta = full_text[len(state["prev_text"]) :]

        # 更新状态
        state["prev_tokens"] = token_ids.copy()
        state["prev_text"] = full_text

        return delta

    def _cleanup_decode_state(self, request_id: str):
        """清理 decode 状态"""
        if request_id in self.decode_states:
            del self.decode_states[request_id]

    def cleanup(self):
        """清理资源"""
        if self.scheduler_socket:
            self.scheduler_socket.close()
        if self.client_socket:
            self.client_socket.close()
