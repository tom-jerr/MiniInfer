# """Scheduler Worker 进程"""

# import logging
# from typing import Optional, Dict, Any, List, Tuple
# from enum import Enum, auto
# from dataclasses import dataclass, field
# import zmq
# import time

# from .base import BaseWorker, WorkerState
# from ..ipc.zmq_channel import create_socket, send_pyobj, recv_pyobj
# from ..ipc.protocol import (
#     MessageType,
#     Request,
#     Response,
#     TokenizeResult,
#     DetokenizeRequest,
# )
# from ...kvcache.kv_cache_manager import KVCacheManager
# from ..seqeunce import Sequence, SequenceStatus

# logger = logging.getLogger(__name__)


# class BatchType(Enum):
#     """Batch 类型"""

#     PREFILL_ONLY = auto()  # 仅 prefill
#     DECODE_ONLY = auto()  # 仅 decode
#     MIXED = auto()  # 混合


# @dataclass
# class ScheduledBatch:
#     """调度后的 Batch"""

#     batch_type: BatchType
#     sequences: List[Sequence] = field(default_factory=list)
#     prefill_seqs: List[Sequence] = field(default_factory=list)
#     decode_seqs: List[Sequence] = field(default_factory=list)


# class SchedulerWorker(BaseWorker):
#     """
#     Scheduler 进程

#     职责:
#     - 从 Tokenizer 接收 tokenize 后的请求
#     - 组织 prefill-only / decode-only / mixed batch
#     - 管理 KV Cache (调用 KVCacheManager)
#     - Post-process 阶段处理 radix cache
#     - 请求完成后释放 KV cache
#     - 将生成的 token ids 发送给 Detokenizer
#     """

#     def __init__(
#         self,
#         # KV Cache 配置
#         max_tokens: int,
#         max_requests: int,
#         max_context_len: int,
#         num_layers: int,
#         num_heads: int,
#         head_dim: int,
#         # Scheduler 配置
#         max_batch_size: int = 32,
#         max_prefill_tokens: int = 4096,
#         enable_prefix_cache: bool = True,
#         schedule_policy: str = "fcfs",  # first-come-first-serve
#         # ZMQ 配置
#         tokenizer_address: str = "",  # 从 Tokenizer 接收
#         detokenizer_address: str = "",  # 发送给 Detokenizer
#         model_runner_address: str = "",  # 与 Model Runner 通信
#         zmq_context: Optional[zmq.Context] = None,
#     ):
#         super().__init__("SchedulerWorker", zmq_context)

#         # Scheduler 配置
#         self.max_batch_size = max_batch_size
#         self.max_prefill_tokens = max_prefill_tokens
#         self.schedule_policy = schedule_policy

#         # ZMQ 地址
#         self.tokenizer_address = tokenizer_address
#         self.detokenizer_address = detokenizer_address
#         self.model_runner_address = model_runner_address

#         # ZMQ sockets
#         self.tokenizer_socket: Optional[zmq.Socket] = None
#         self.detokenizer_socket: Optional[zmq.Socket] = None
#         self.model_runner_socket: Optional[zmq.Socket] = None

#         # KV Cache Manager
#         self.kv_cache_manager: Optional[KVCacheManager] = None
#         self.kv_cache_config = {
#             "max_tokens": max_tokens,
#             "max_requests": max_requests,
#             "max_context_len": max_context_len,
#             "num_layers": num_layers,
#             "num_heads": num_heads,
#             "head_dim": head_dim,
#             "enable_prefix_cache": enable_prefix_cache,
#         }

#         # 请求队列
#         self.waiting_queue: List[Sequence] = []
#         self.running_queue: List[Sequence] = []
#         self.finished_queue: List[Sequence] = []

#         # 请求映射
#         self.request_map: Dict[str, Sequence] = {}  # request_id -> Sequence
#         self.seq_to_req_idx: Dict[int, int] = {}  # seq_id -> req_idx in KVCacheManager

#     def setup(self):
#         """初始化"""
#         # 初始化 KV Cache Manager
#         logger.info("Initializing KVCacheManager...")
#         self.kv_cache_manager = KVCacheManager(**self.kv_cache_config)

#         # 设置 Tokenizer socket (PULL 接收)
#         self.tokenizer_socket = create_socket(
#             self.zmq_context, zmq.PULL, self.tokenizer_address, bind=True
#         )

#         # 设置 Detokenizer socket (PUSH 发送)
#         self.detokenizer_socket = create_socket(
#             self.zmq_context, zmq.PUSH, self.detokenizer_address, bind=False
#         )

#         logger.info("SchedulerWorker setup complete")

#     def process(self) -> bool:
#         """处理一轮"""
#         # 1. 接收新请求
#         self._receive_requests()

#         # 2. 调度 Batch
#         batch = self._schedule_batch()

#         if batch is None or len(batch.sequences) == 0:
#             time.sleep(0.001)  # 避免空转
#             return True

#         # 3. 分配 KV Cache
#         if not self._allocate_kv_cache(batch):
#             logger.warning("Failed to allocate KV cache")
#             return True

#         # 4. 执行模型推理 (这里简化，实际需要与 ModelRunner 通信)
#         # self._run_model(batch)

#         # 5. Post-process
#         self._post_process(batch)

#         return True

#     def _receive_requests(self):
#         """接收新请求"""
#         while True:
#             result = recv_pyobj(self.tokenizer_socket, timeout=0)
#             if result is None:
#                 break

#             _, data = result
#             if (
#                 isinstance(data, Request)
#                 and data.msg_type == MessageType.TOKENIZE_RESULT
#             ):
#                 self._handle_tokenize_result(data)

#     def _handle_tokenize_result(self, request: Request):
#         """处理 tokenize 结果"""
#         result: TokenizeResult = request.payload

#         # 创建 Sequence
#         seq = Sequence(result.token_ids, result.sampling_params)

#         # 添加到等待队列
#         self.waiting_queue.append(seq)
#         self.request_map[result.request_id] = seq

#         logger.debug(
#             f"Added request {result.request_id} to waiting queue, "
#             f"{len(result.token_ids)} tokens"
#         )

#     def _schedule_batch(self) -> Optional[ScheduledBatch]:
#         """
#         调度 Batch

#         策略:
#         1. 优先处理 decode (延迟敏感)
#         2. 然后处理 prefill
#         3. 支持混合 batch
#         """
#         if not self.waiting_queue and not self.running_queue:
#             return None

#         batch = ScheduledBatch(batch_type=BatchType.MIXED)

#         # 1. 添加 decode 请求 (running queue 中的请求)
#         decode_budget = self.max_batch_size
#         for seq in self.running_queue[:decode_budget]:
#             if seq.status == SequenceStatus.RUNNING:
#                 batch.decode_seqs.append(seq)
#                 batch.sequences.append(seq)

#         # 2. 添加 prefill 请求
#         prefill_budget = self.max_batch_size - len(batch.decode_seqs)
#         prefill_tokens = 0

#         new_waiting = []
#         for seq in self.waiting_queue:
#             if len(batch.prefill_seqs) >= prefill_budget:
#                 new_waiting.append(seq)
#                 continue

#             seq_tokens = seq.num_tokens - seq.num_cached_tokens
#             if prefill_tokens + seq_tokens > self.max_prefill_tokens:
#                 new_waiting.append(seq)
#                 continue

#             # 检查 KV cache 空间
#             if not self.kv_cache_manager.can_allocate(seq_tokens):
#                 new_waiting.append(seq)
#                 continue

#             batch.prefill_seqs.append(seq)
#             batch.sequences.append(seq)
#             prefill_tokens += seq_tokens

#         self.waiting_queue = new_waiting

#         # 确定 batch 类型
#         if batch.prefill_seqs and not batch.decode_seqs:
#             batch.batch_type = BatchType.PREFILL_ONLY
#         elif batch.decode_seqs and not batch.prefill_seqs:
#             batch.batch_type = BatchType.DECODE_ONLY
#         else:
#             batch.batch_type = BatchType.MIXED

#         return batch

#     def _allocate_kv_cache(self, batch: ScheduledBatch) -> bool:
#         """为 batch 分配 KV Cache"""
#         for seq in batch.prefill_seqs:
#             # 分配请求槽位
#             req_idx = self.kv_cache_manager.alloc_request()
#             if req_idx is None:
#                 logger.error(f"Failed to allocate request slot for seq {seq.seq_id}")
#                 return False

#             self.seq_to_req_idx[seq.seq_id] = req_idx

#             # 分配 KV cache
#             num_new_tokens = seq.num_tokens
#             kv_indices, num_cached = self.kv_cache_manager.alloc_for_request(
#                 req_idx=req_idx,
#                 token_ids=seq.token_ids,
#                 num_new_tokens=num_new_tokens,
#             )

#             if kv_indices is None and num_cached < num_new_tokens:
#                 logger.error(f"Failed to allocate KV cache for seq {seq.seq_id}")
#                 return False

#             seq.num_cached_tokens = num_cached
#             seq.status = SequenceStatus.RUNNING
#             self.running_queue.append(seq)

#         # Decode 请求只需要分配 1 个 token
#         for seq in batch.decode_seqs:
#             req_idx = self.seq_to_req_idx.get(seq.seq_id)
#             if req_idx is None:
#                 continue

#             kv_indices = self.kv_cache_manager.alloc_tokens(1)
#             if kv_indices is not None:
#                 # 更新映射
#                 token_pos = seq.num_tokens
#                 self.kv_cache_manager.request_pool.write(
#                     req_idx, slice(token_pos, token_pos + 1), kv_indices
#                 )

#         return True

#     def _post_process(self, batch: ScheduledBatch):
#         """
#         Post-process 阶段

#         - 处理完成的请求
#         - 更新 radix cache
#         - 释放资源
#         """
#         finished_seqs = []

#         for seq in batch.sequences:
#             # 检查是否完成 (简化判断)
#             if seq.is_finished:
#                 finished_seqs.append(seq)

#         # 处理完成的请求
#         for seq in finished_seqs:
#             req_idx = self.seq_to_req_idx.get(seq.seq_id)
#             if req_idx is not None:
#                 # 释放 KV cache 并缓存到 radix tree
#                 self.kv_cache_manager.release_request(
#                     req_idx=req_idx,
#                     token_ids=seq.token_ids,
#                     num_tokens=seq.num_tokens,
#                     cache_to_radix=True,
#                 )
#                 del self.seq_to_req_idx[seq.seq_id]

#             # 从 running queue 移除
#             if seq in self.running_queue:
#                 self.running_queue.remove(seq)

#             # 添加到 finished queue
#             self.finished_queue.append(seq)

#             # 发送给 Detokenizer
#             self._send_to_detokenizer(seq)

#     def _send_to_detokenizer(self, seq: Sequence):
#         """发送完成的请求给 Detokenizer"""
#         # 找到 request_id
#         request_id = None
#         for rid, s in self.request_map.items():
#             if s.seq_id == seq.seq_id:
#                 request_id = rid
#                 break

#         if request_id is None:
#             return

#         detokenize_request = DetokenizeRequest(
#             request_id=request_id,
#             token_ids=seq.token_ids,
#         )

#         request = Request(
#             request_id=request_id,
#             msg_type=MessageType.DETOKENIZE_REQUEST,
#             payload=detokenize_request,
#         )

#         send_pyobj(self.detokenizer_socket, request)
#         logger.debug(f"Sent request {request_id} to detokenizer")

#     def cleanup(self):
#         """清理资源"""
#         if self.tokenizer_socket:
#             self.tokenizer_socket.close()
#         if self.detokenizer_socket:
#             self.detokenizer_socket.close()
#         if self.model_runner_socket:
#             self.model_runner_socket.close()
