"""Scheduler Worker 进程（EngineCore）。

职责：
- 从 Tokenizer 接收 TokenizeResult（PULL）；
- 复用单进程 `LLMEngine` 的全部推理能力（Scheduler + ModelRunner + KVCache
  + overlap + CUDA graph），通过 `engine.add_request(token_ids)` 注入请求、
  `engine.step_overlap()` 驱动推理；
- 把每步产出的 token（全量 output_token_ids）作为 TokenChunk 发给 Detokenizer。

req_id 映射：客户端 request_id 与 LLMEngine 内部 Req.req_id 不同，这里维护
`engine_req_id -> client_request_id` 映射，输出时还原。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import zmq

from miniinfer.utils import get_logger

from ..ipc.protocol import MessageType, Request, TokenChunk, TokenizeResult
from ..ipc.zmq_channel import create_socket, recv_pyobj, send_pyobj
from .base import BaseWorker

logger = get_logger(__name__)


class SchedulerWorker(BaseWorker):
  def __init__(
    self,
    model_path: str,
    tokenizer_address: str,  # 从 Tokenizer 接收（PULL bind）
    detokenizer_address: str,  # 发送给 Detokenizer（PUSH connect）
    engine_kwargs: Optional[Dict[str, Any]] = None,
    zmq_context: Optional[zmq.Context] = None,
  ):
    super().__init__("SchedulerWorker", zmq_context)
    self.model_path = model_path
    self.tokenizer_address = tokenizer_address
    self.detokenizer_address = detokenizer_address
    self.engine_kwargs = engine_kwargs or {}

    self.engine = None
    self.tokenizer_socket: Optional[zmq.Socket] = None
    self.detokenizer_socket: Optional[zmq.Socket] = None
    # engine_req_id -> client_request_id
    self.req_id_map: Dict[int, int] = {}

  def setup(self) -> None:
    # 延迟导入，避免父进程 fork/spawn 时过早初始化 CUDA。
    from miniinfer.engine.llm_engine import LLMEngine

    logger.info(f"Initializing LLMEngine (model={self.model_path})...")
    # EngineCore 始终走单进程模式（本进程内即 GPU 推理核）。
    self.engine = LLMEngine(
      self.model_path, use_multiprocess=False, **self.engine_kwargs
    )
    self.engine.start()
    logger.info("LLMEngine started")

    self.tokenizer_socket = create_socket(
      self.zmq_context, zmq.PULL, self.tokenizer_address, bind=True
    )
    self.detokenizer_socket = create_socket(
      self.zmq_context, zmq.PUSH, self.detokenizer_address, bind=False
    )
    logger.info("SchedulerWorker setup complete")

  def process(self) -> bool:
    # 1. 尽量收完新请求（非阻塞）
    self._receive_requests()

    # 2. 有活就推一步；没活就短暂休眠避免空转
    if self.engine.scheduler.has_unfinished():
      step_out = self.engine.step_overlap()
      self._dispatch_outputs(step_out.outputs)
    else:
      time.sleep(0.001)
    return True

  def _receive_requests(self) -> None:
    while True:
      result = recv_pyobj(self.tokenizer_socket, timeout=0)
      if result is None:
        break
      _, data = result
      if isinstance(data, Request):
        if data.msg_type == MessageType.TOKENIZE_RESULT:
          self._handle_tokenize_result(data)
        elif data.msg_type == MessageType.SHUTDOWN:
          self._shutdown_event.set()

  def _handle_tokenize_result(self, request: Request) -> None:
    result: TokenizeResult = request.payload
    # add_request 收到 list[int] 时跳过 tokenize，直接用 token_ids。
    engine_req_id = self.engine.add_request(
      result.token_ids, result.sampling_params
    )
    self.req_id_map[engine_req_id] = result.request_id
    logger.debug(
      f"Added request client={result.request_id} engine={engine_req_id} "
      f"({len(result.token_ids)} tokens)"
    )

  def _dispatch_outputs(self, outputs) -> None:
    for o in outputs:
      client_rid = self.req_id_map.get(o.request_id)
      if client_rid is None:
        # 未知 req_id（可能是 add_request 之外的路径），跳过。
        continue
      send_pyobj(
        self.detokenizer_socket,
        Request(
          request_id=client_rid,
          msg_type=MessageType.TOKEN_CHUNK,
          payload=TokenChunk(
            request_id=client_rid,
            output_token_ids=list(o.output_token_ids),
            finished=o.finished,
          ),
        ),
      )
      if o.finished:
        self.req_id_map.pop(o.request_id, None)

  def cleanup(self) -> None:
    # 通知 detokenizer 关闭（尽力而为）。
    if self.detokenizer_socket is not None:
      try:
        send_pyobj(
          self.detokenizer_socket,
          Request(request_id=-1, msg_type=MessageType.SHUTDOWN),
        )
      except Exception:
        pass
    if self.engine is not None:
      try:
        self.engine.stop()
      except Exception as e:
        logger.exception(f"engine.stop error: {e}")
    if self.tokenizer_socket:
      self.tokenizer_socket.close()
    if self.detokenizer_socket:
      self.detokenizer_socket.close()
