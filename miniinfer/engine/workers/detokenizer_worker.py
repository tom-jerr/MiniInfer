"""Detokenizer Worker 进程。

职责：
- 从 Scheduler 接收每步生成的 TokenChunk（PULL）；
- 增量 detokenize（按全量 token_ids 解码，取与上次的差量作为 delta_text）；
- 把 GenerateResponse 发送给客户端（PUSH；v1 单客户端，多客户端路由待扩展）。
"""
from __future__ import annotations

from typing import Dict, Optional

import zmq
from transformers import AutoTokenizer

from miniinfer.utils import get_logger

from ..ipc.protocol import GenerateResponse, MessageType, Request, TokenChunk
from ..ipc.zmq_channel import create_socket, recv_pyobj, send_pyobj
from .base import BaseWorker

logger = get_logger(__name__)


class DetokenizerWorker(BaseWorker):
  def __init__(
    self,
    model_path: str,
    scheduler_address: str,  # 从 Scheduler 接收（PULL bind）
    client_address: str,  # 发送给客户端（PUSH bind，v1 单客户端）
    zmq_context: Optional[zmq.Context] = None,
  ):
    super().__init__("DetokenizerWorker", zmq_context)
    self.model_path = model_path
    self.scheduler_address = scheduler_address
    self.client_address = client_address

    self.tokenizer: Optional[AutoTokenizer] = None
    self.scheduler_socket: Optional[zmq.Socket] = None
    self.client_socket: Optional[zmq.Socket] = None
    # request_id -> {"prev_text": str}
    self.decode_states: Dict[int, dict] = {}

  def setup(self) -> None:
    logger.info(f"Loading tokenizer from {self.model_path}")
    self.tokenizer = AutoTokenizer.from_pretrained(
      self.model_path, use_fast=True, trust_remote_code=True
    )

    self.scheduler_socket = create_socket(
      self.zmq_context, zmq.PULL, self.scheduler_address, bind=True
    )
    self.client_socket = create_socket(
      self.zmq_context, zmq.PUSH, self.client_address, bind=True
    )
    logger.info("DetokenizerWorker setup complete")

  def process(self) -> bool:
    result = recv_pyobj(self.scheduler_socket, timeout=100)
    if result is None:
      return True

    _, data = result
    if isinstance(data, Request):
      if data.msg_type == MessageType.TOKEN_CHUNK:
        self._handle_token_chunk(data)
      elif data.msg_type == MessageType.SHUTDOWN:
        self._shutdown_event.set()
    return True

  def _handle_token_chunk(self, request: Request) -> None:
    chunk: TokenChunk = request.payload
    delta = self._incremental_detokenize(chunk.request_id, chunk.output_token_ids)

    send_pyobj(
      self.client_socket,
      Request(
        request_id=chunk.request_id,
        msg_type=MessageType.GENERATE_RESPONSE,
        payload=GenerateResponse(
          request_id=chunk.request_id,
          delta_text=delta,
          finished=chunk.finished,
          output_token_ids=list(chunk.output_token_ids),
        ),
      ),
    )
    if chunk.finished:
      self._cleanup_decode_state(chunk.request_id)

  def _incremental_detokenize(self, request_id: int, token_ids: list) -> str:
    """增量 detokenize：按全量 token 解码，取差量。"""
    state = self.decode_states.get(request_id)
    if state is None:
      state = {"prev_text": ""}
      self.decode_states[request_id] = state

    full_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
    delta = full_text[len(state["prev_text"]) :]
    state["prev_text"] = full_text
    return delta

  def _cleanup_decode_state(self, request_id: int) -> None:
    self.decode_states.pop(request_id, None)

  def cleanup(self) -> None:
    if self.scheduler_socket:
      self.scheduler_socket.close()
    if self.client_socket:
      self.client_socket.close()
