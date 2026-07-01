"""多进程推理引擎驱动（client 侧）。

按 `ipc/` + `workers/` 的思路实现：spawn 三个 worker 进程——Tokenizer、
Scheduler（EngineCore，复用单进程 `LLMEngine`）、Detokenizer——用 ZMQ (ipc://)
串成流水线。本类是 client：把 prompt 发给 Tokenizer、从 Detokenizer 收增量文本。

对用户暴露与 `LLMEngine` 一致的 `generate` / `stream_generate` / `start` / `stop`
/ 上下文管理器接口，可直接替换单进程引擎。

拓扑与 req_id 映射见 `ipc/protocol.py` 与 `workers/scheduler_worker.py`。
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Dict, Generator, List, Optional, Union

import zmq

from miniinfer.engine.ipc.protocol import (
  GenerateRequest,
  MessageType,
  Request,
)
from miniinfer.engine.ipc.zmq_channel import (
  create_socket,
  make_ipc_address,
  recv_pyobj,
  send_pyobj,
)
from miniinfer.engine.workers.base import worker_process_entry
from miniinfer.engine.workers.detokenizer_worker import DetokenizerWorker
from miniinfer.engine.workers.scheduler_worker import SchedulerWorker
from miniinfer.engine.workers.tokenizer_worker import TokenizerWorker
from miniinfer.utils import get_logger
from miniinfer.utils.sampling_params import SamplingParams

logger = get_logger(__name__)


class MultiProcessEngine:
  """多进程推理引擎（Tokenizer / EngineCore / Detokenizer 三进程）。"""

  def __init__(
    self,
    model: str,
    *,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    base_ipc_dir: Optional[str] = None,
  ):
    self.model_path = model
    self.engine_kwargs = engine_kwargs or {}
    self._base_ipc_dir = base_ipc_dir or f"/tmp/miniinfer/{os.getpid()}"
    self._mp_ctx = mp.get_context("spawn")  # spawn：避免 CUDA fork 问题
    self._processes: List[mp.Process] = []
    self._zmq_ctx: Optional[zmq.Context] = None
    self._req_socket: Optional[zmq.Socket] = None  # DEALER -> tokenizer
    self._resp_socket: Optional[zmq.Socket] = None  # PULL <- detokenizer
    self._addresses: Dict[str, str] = {}
    self._started = False
    self._req_counter = 0

  # ------------------------------------------------------------------
  # 生命周期
  # ------------------------------------------------------------------
  def start(self) -> None:
    if self._started:
      return

    a = {
      "client_to_tokenizer": make_ipc_address("client_to_tokenizer", self._base_ipc_dir),
      "tokenizer_to_scheduler": make_ipc_address("tok2sched", self._base_ipc_dir),
      "scheduler_to_detokenizer": make_ipc_address("sched2detok", self._base_ipc_dir),
      "detokenizer_to_client": make_ipc_address("detok2client", self._base_ipc_dir),
    }
    self._addresses = a

    # 1. 启动 worker 进程（bind 端先起：tokenizer ROUTER / scheduler PULL / detokenizer PULL+PUSH）
    self._processes = [
      self._mp_ctx.Process(
        target=worker_process_entry,
        args=(TokenizerWorker, self.model_path, a["client_to_tokenizer"], a["tokenizer_to_scheduler"]),
        name="TokenizerWorker",
        daemon=True,
      ),
      self._mp_ctx.Process(
        target=worker_process_entry,
        args=(DetokenizerWorker, self.model_path, a["scheduler_to_detokenizer"], a["detokenizer_to_client"]),
        name="DetokenizerWorker",
        daemon=True,
      ),
      self._mp_ctx.Process(
        target=worker_process_entry,
        args=(SchedulerWorker, self.model_path, a["tokenizer_to_scheduler"], a["scheduler_to_detokenizer"], self.engine_kwargs),
        name="SchedulerWorker",
        daemon=True,
      ),
    ]
    for p in self._processes:
      p.start()
    logger.info(
      f"Started 3 worker processes: {[p.name for p in self._processes]}. "
      f"EngineCore will load model {self.model_path} (this takes a while)..."
    )

    # 2. client 侧 socket：ZMQ 允许 connect 早于 bind（会自动重连）。
    self._zmq_ctx = zmq.Context.instance()
    self._req_socket = create_socket(self._zmq_ctx, zmq.DEALER, a["client_to_tokenizer"], bind=False)
    self._resp_socket = create_socket(self._zmq_ctx, zmq.PULL, a["detokenizer_to_client"], bind=False)
    self._started = True

  def stop(self) -> None:
    if not self._started:
      return
    # 给 worker 发 SHUTDOWN（尽力），再 terminate 进程。
    try:
      if self._req_socket is not None:
        send_pyobj(self._req_socket, Request(request_id=-1, msg_type=MessageType.SHUTDOWN))
    except Exception:
      pass
    for p in self._processes:
      p.terminate()
    for p in self._processes:
      p.join(timeout=5)
      if p.is_alive():
        p.kill()
    if self._req_socket:
      self._req_socket.close()
    if self._resp_socket:
      self._resp_socket.close()
    self._processes = []
    self._started = False
    # 清理本实例的 ipc socket 目录。
    try:
      import shutil

      if os.path.isdir(self._base_ipc_dir):
        shutil.rmtree(self._base_ipc_dir, ignore_errors=True)
    except Exception:
      pass
    logger.info("MultiProcessEngine stopped")

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.stop()
    return False

  # ------------------------------------------------------------------
  # 请求发送 / 响应接收
  # ------------------------------------------------------------------
  def _next_request_id(self) -> int:
    rid = self._req_counter
    self._req_counter += 1
    return rid

  def _send_request(self, prompt: str, sampling_params: SamplingParams) -> int:
    rid = self._next_request_id()
    send_pyobj(
      self._req_socket,
      Request(
        request_id=rid,
        msg_type=MessageType.GENERATE_REQUEST,
        payload=GenerateRequest(request_id=rid, prompt=prompt, sampling_params=sampling_params),
      ),
    )
    return rid

  def _recv_response(self, timeout_ms: int = 600000):
    """收一个响应；超时抛 TimeoutError。

    默认 10 分钟，足够覆盖大模型在 EngineCore 进程里的加载 + 首个 token 时间。
    """
    res = recv_pyobj(self._resp_socket, timeout=timeout_ms)
    if res is None:
      raise TimeoutError("Timed out waiting for detokenizer response")
    _, data = res
    return data

  # ------------------------------------------------------------------
  # 公共 API（与 LLMEngine 对齐）
  # ------------------------------------------------------------------
  def generate(
    self,
    prompts: Union[str, List[str]],
    sampling_params: Union[SamplingParams, List[SamplingParams], None] = None,
    use_tqdm: bool = False,
  ) -> List[Dict[str, Any]]:
    if not self._started:
      self.start()

    if isinstance(prompts, str):
      prompts = [prompts]
    if sampling_params is None:
      sampling_params = SamplingParams()
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)

    results: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    pending = set()
    for p, sp in zip(prompts, sampling_params):
      rid = self._send_request(p, sp)
      results[rid] = {"text": "", "token_ids": [], "request_id": rid}
      order.append(rid)
      pending.add(rid)

    while pending:
      data = self._recv_response()
      if not isinstance(data, Request) or data.msg_type != MessageType.GENERATE_RESPONSE:
        continue
      gr = data.payload
      if gr.request_id not in pending:
        continue
      entry = results[gr.request_id]
      entry["text"] += gr.delta_text
      if gr.output_token_ids:
        entry["token_ids"] = gr.output_token_ids  # 全量，最后一条为准
      if gr.finished:
        pending.discard(gr.request_id)

    return [results[rid] for rid in order]

  def stream_generate(
    self,
    prompts: Union[str, List[str]],
    sampling_params: Union[SamplingParams, List[SamplingParams], None] = None,
  ) -> Generator[Any, None, None]:
    """流式生成：每收到一段 delta 就 yield 一个轻量输出对象。"""
    from dataclasses import dataclass

    if not self._started:
      self.start()
    if isinstance(prompts, str):
      prompts = [prompts]
    if sampling_params is None:
      sampling_params = SamplingParams()
    if not isinstance(sampling_params, list):
      sampling_params = [sampling_params] * len(prompts)

    @dataclass
    class StreamOutput:
      request_id: int
      delta_text: str
      finished: bool

    pending = set()
    for p, sp in zip(prompts, sampling_params):
      rid = self._send_request(p, sp)
      pending.add(rid)

    while pending:
      data = self._recv_response()
      if not isinstance(data, Request) or data.msg_type != MessageType.GENERATE_RESPONSE:
        continue
      gr = data.payload
      if gr.request_id not in pending:
        continue
      yield StreamOutput(request_id=gr.request_id, delta_text=gr.delta_text, finished=gr.finished)
      if gr.finished:
        pending.discard(gr.request_id)
