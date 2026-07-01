"""IPC 消息协议定义。

多进程推理引擎的进程间通信基于 ZMQ + pickle。本模块定义消息类型与载荷
dataclass，供 Tokenizer / Scheduler(EngineCore) / Detokenizer 三个 worker
以及客户端（MultiProcessEngine）共用。

拓扑（同机器用 Unix Domain Socket ipc://）：

    Client ──DEALER──▶ ROUTER ── Tokenizer ──PUSH──▶ PULL ── Scheduler(EngineCore)
                                                                     │
    Client ◀──PULL─── PUSH ── Detokenizer ◀──PULL── PUSH ────────────┘

请求标识：客户端生成 request_id（int），随请求流经全链路；Scheduler 内部
把它与 LLMEngine 的 Req.req_id 做映射，输出时再映射回 client request_id。
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


class MessageType(Enum):
  """消息类型。"""

  # Client -> Tokenizer
  GENERATE_REQUEST = auto()
  # Tokenizer -> Scheduler
  TOKENIZE_RESULT = auto()
  # Scheduler -> Detokenizer
  TOKEN_CHUNK = auto()
  # Detokenizer -> Client
  GENERATE_RESPONSE = auto()

  # 控制消息
  SHUTDOWN = auto()
  HEALTH_CHECK = auto()
  HEALTH_RESPONSE = auto()


@dataclass
class Request:
  """请求信封。"""

  request_id: int
  msg_type: MessageType
  payload: Any = None
  timestamp: float = field(default_factory=time.time)

  def serialize(self) -> bytes:
    return pickle.dumps(self)

  @classmethod
  def deserialize(cls, data: bytes) -> "Request":
    return pickle.loads(data)


@dataclass
class Response:
  """响应信封。"""

  request_id: int
  msg_type: MessageType
  payload: Any = None
  success: bool = True
  error: Optional[str] = None
  timestamp: float = field(default_factory=time.time)

  def serialize(self) -> bytes:
    return pickle.dumps(self)

  @classmethod
  def deserialize(cls, data: bytes) -> "Response":
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# 载荷 dataclass
# ---------------------------------------------------------------------------
@dataclass
class GenerateRequest:
  """Client -> Tokenizer：生成请求。"""

  request_id: int
  prompt: str
  sampling_params: Any  # SamplingParams
  stream: bool = False


@dataclass
class TokenizeResult:
  """Tokenizer -> Scheduler：tokenize 结果。"""

  request_id: int
  token_ids: List[int]
  sampling_params: Any  # SamplingParams


@dataclass
class TokenChunk:
  """Scheduler -> Detokenizer：一步生成的 token（全量 token_ids + finished）。"""

  request_id: int
  output_token_ids: List[int]  # 截至本步的全量输出 token
  finished: bool = False


@dataclass
class GenerateResponse:
  """Detokenizer -> Client：增量文本响应。"""

  request_id: int
  delta_text: str
  finished: bool = False
  output_token_ids: List[int] = field(default_factory=list)  # 截至本步全量 token
