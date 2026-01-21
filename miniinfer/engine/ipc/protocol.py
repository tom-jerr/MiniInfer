# """IPC 消息协议定义"""

# from dataclasses import dataclass, field
# from enum import Enum, auto
# from typing import Any, Optional, List
# import pickle
# import time


# class MessageType(Enum):
#     """消息类型"""

#     # Tokenizer -> Scheduler
#     TOKENIZE_RESULT = auto()

#     # Scheduler -> Tokenizer
#     TOKENIZE_REQUEST = auto()

#     # Scheduler -> Detokenizer
#     DETOKENIZE_REQUEST = auto()

#     # Detokenizer -> Scheduler
#     DETOKENIZE_RESULT = auto()

#     # Client -> Tokenizer (external requests)
#     GENERATE_REQUEST = auto()
#     GENERATE_RESPONSE = auto()

#     # Control messages
#     SHUTDOWN = auto()
#     HEALTH_CHECK = auto()
#     HEALTH_RESPONSE = auto()

#     # Batch signals
#     BATCH_READY = auto()
#     BATCH_COMPLETE = auto()


# @dataclass
# class Request:
#     """请求基类"""

#     request_id: str
#     msg_type: MessageType
#     payload: Any = None
#     timestamp: float = field(default_factory=time.time)

#     def serialize(self) -> bytes:
#         return pickle.dumps(self)

#     @classmethod
#     def deserialize(cls, data: bytes) -> "Request":
#         return pickle.loads(data)


# @dataclass
# class Response:
#     """响应基类"""

#     request_id: str
#     msg_type: MessageType
#     payload: Any = None
#     success: bool = True
#     error: Optional[str] = None
#     timestamp: float = field(default_factory=time.time)

#     def serialize(self) -> bytes:
#         return pickle.dumps(self)

#     @classmethod
#     def deserialize(cls, data: bytes) -> "Response":
#         return pickle.loads(data)


# @dataclass
# class TokenizeRequest:
#     """Tokenize 请求"""

#     request_id: str
#     prompt: str
#     sampling_params: Any  # SamplingParams


# @dataclass
# class TokenizeResult:
#     """Tokenize 结果"""

#     request_id: str
#     token_ids: List[int]
#     sampling_params: Any


# @dataclass
# class DetokenizeRequest:
#     """Detokenize 请求"""

#     request_id: str
#     token_ids: List[int]
#     skip_special_tokens: bool = True


# @dataclass
# class DetokenizeResult:
#     """Detokenize 结果"""

#     request_id: str
#     text: str


# @dataclass
# class GenerateRequest:
#     """生成请求"""

#     request_id: str
#     prompt: str
#     sampling_params: Any
#     stream: bool = False


# @dataclass
# class GenerateResponse:
#     """生成响应"""

#     request_id: str
#     text: str
#     token_ids: List[int]
#     finished: bool = False
#     metrics: Optional[dict] = None
