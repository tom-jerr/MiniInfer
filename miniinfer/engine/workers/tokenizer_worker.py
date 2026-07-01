"""Tokenizer Worker 进程。

职责：
- 接收客户端生成请求（ROUTER，支持多客户端 identity）；
- 将 prompt tokenize 为 token ids；
- 把 TokenizeResult 发送给 Scheduler（PUSH）。

request_id 由客户端生成，原样透传，作为全链路路由键。
"""
from __future__ import annotations

from typing import Dict, Optional

import zmq
from transformers import AutoTokenizer

from miniinfer.utils import get_logger

from ..ipc.protocol import GenerateRequest, MessageType, Request, TokenizeResult
from ..ipc.zmq_channel import create_socket, recv_pyobj, send_pyobj
from .base import BaseWorker

logger = get_logger(__name__)


class TokenizerWorker(BaseWorker):
  def __init__(
    self,
    model_path: str,
    client_address: str,  # 接收客户端请求（ROUTER bind）
    scheduler_address: str,  # 发送给 Scheduler（PUSH connect）
    zmq_context: Optional[zmq.Context] = None,
  ):
    super().__init__("TokenizerWorker", zmq_context)
    self.model_path = model_path
    self.client_address = client_address
    self.scheduler_address = scheduler_address

    self.tokenizer: Optional[AutoTokenizer] = None
    self.client_socket: Optional[zmq.Socket] = None
    self.scheduler_socket: Optional[zmq.Socket] = None
    # request_id -> client identity（多客户端时用于回路由由；当前 v1 单客户端）
    self.pending_requests: Dict[int, bytes] = {}

  def setup(self) -> None:
    logger.info(f"Loading tokenizer from {self.model_path}")
    self.tokenizer = AutoTokenizer.from_pretrained(
      self.model_path, use_fast=True, trust_remote_code=True
    )

    self.client_socket = create_socket(
      self.zmq_context, zmq.ROUTER, self.client_address, bind=True, identity="tokenizer"
    )
    self.scheduler_socket = create_socket(
      self.zmq_context, zmq.PUSH, self.scheduler_address, bind=False
    )
    logger.info("TokenizerWorker setup complete")

  def process(self) -> bool:
    result = recv_pyobj(self.client_socket, timeout=100)  # 100ms 心跳窗口
    if result is None:
      return True

    identity, data = result
    if isinstance(data, Request):
      self._handle_request(identity, data)
    return True

  def _handle_request(self, identity: bytes, request: Request) -> None:
    if request.msg_type == MessageType.GENERATE_REQUEST:
      self._handle_generate_request(identity, request)
    elif request.msg_type == MessageType.HEALTH_CHECK:
      from ..ipc.protocol import Response

      send_pyobj(
        self.client_socket,
        Response(
          request_id=request.request_id,
          msg_type=MessageType.HEALTH_RESPONSE,
          payload={"state": self.state.name},
        ),
        identity=identity,
      )
    elif request.msg_type == MessageType.SHUTDOWN:
      self._shutdown_event.set()

  def _handle_generate_request(self, identity: bytes, request: Request) -> None:
    gen_request: GenerateRequest = request.payload
    token_ids = self.tokenizer.encode(gen_request.prompt)
    self.pending_requests[gen_request.request_id] = identity

    send_pyobj(
      self.scheduler_socket,
      Request(
        request_id=gen_request.request_id,
        msg_type=MessageType.TOKENIZE_RESULT,
        payload=TokenizeResult(
          request_id=gen_request.request_id,
          token_ids=token_ids,
          sampling_params=gen_request.sampling_params,
        ),
      ),
    )
    logger.debug(f"Tokenized request {gen_request.request_id}: {len(token_ids)} tokens")

  def cleanup(self) -> None:
    if self.client_socket:
      self.client_socket.close()
    if self.scheduler_socket:
      self.scheduler_socket.close()
