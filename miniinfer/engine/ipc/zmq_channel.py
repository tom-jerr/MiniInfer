"""ZMQ 通信模块 - 直接使用 pyzmq。

进程间通信拓扑（见 protocol.py）。同机器用 Unix Domain Socket (ipc://)，
跨机器可用 TCP (tcp://)。提供 socket 工厂、序列化、收发与多 socket 轮询。
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import zmq

from miniinfer.utils import get_logger

logger = get_logger(__name__)


# ============== 地址生成工具 ==============
def make_ipc_address(name: str, base_dir: str = "/tmp/miniinfer") -> str:
  """生成 IPC 地址 (Unix Domain Socket)。"""
  os.makedirs(base_dir, exist_ok=True)
  return f"ipc://{base_dir}/{name}.sock"


def make_tcp_address(host: str = "127.0.0.1", port: int = 5555) -> str:
  """生成 TCP 地址。"""
  return f"tcp://{host}:{port}"


def cleanup_ipc_socket(address: str) -> None:
  """清理 IPC socket 文件。"""
  if address.startswith("ipc://"):
    sock_path = address[6:]
    if os.path.exists(sock_path):
      try:
        os.unlink(sock_path)
      except FileNotFoundError:
        pass


# ============== 序列化工具 ==============
def serialize(obj: Any) -> bytes:
  return pickle.dumps(obj)


def deserialize(data: bytes) -> Any:
  return pickle.loads(data)


# ============== Socket 工厂 ==============
def create_socket(
  ctx: zmq.Context,
  socket_type: int,
  address: str,
  bind: bool = False,
  identity: Optional[str] = None,
) -> zmq.Socket:
  """创建并配置 ZMQ socket。

  Args:
      ctx: ZMQ Context
      socket_type: zmq.PUSH / zmq.PULL / zmq.ROUTER / zmq.DEALER 等
      address: ipc:// 或 tcp:// 地址
      bind: True=bind（服务端），False=connect（客户端）
      identity: ROUTER/DEALER 的 socket 标识
  """
  socket = ctx.socket(socket_type)
  if identity:
    socket.setsockopt_string(zmq.IDENTITY, identity)
  socket.setsockopt(zmq.LINGER, 0)  # 关闭时不等待，避免挂住

  if bind:
    cleanup_ipc_socket(address)
    socket.bind(address)
    logger.debug(f"Bound {socket_type} to {address}")
  else:
    socket.connect(address)
    logger.debug(f"Connected {socket_type} to {address}")
  return socket


# ============== 发送/接收 ==============
def send_pyobj(socket: zmq.Socket, obj: Any, identity: Optional[bytes] = None) -> None:
  """发送 Python 对象。

  Args:
      identity: 目标 identity（仅 ROUTER socket 需要）
  """
  data = serialize(obj)
  if identity is not None:
    # ROUTER socket: [identity, '', data]
    socket.send_multipart([identity, b"", data])
  else:
    socket.send(data)


def recv_pyobj(
  socket: zmq.Socket,
  timeout: Optional[int] = None,
) -> Optional[Tuple[Optional[bytes], Any]]:
  """接收 Python 对象。

  Args:
      timeout: 超时毫秒，None=阻塞，0=非阻塞

  Returns:
      (identity, obj)；identity 仅 ROUTER 有效；超时返回 None
  """
  if timeout is not None:
    if not socket.poll(timeout, zmq.POLLIN):
      return None

  socket_type = socket.getsockopt(zmq.TYPE)
  if socket_type == zmq.ROUTER:
    frames = socket.recv_multipart()
    identity = frames[0]
    data = frames[-1]
  else:
    identity = None
    data = socket.recv()

  obj = deserialize(data)
  return identity, obj


def recv_pyobj_nowait(socket: zmq.Socket) -> Optional[Tuple[Optional[bytes], Any]]:
  """非阻塞接收。"""
  return recv_pyobj(socket, timeout=0)


# ============== Poller 封装 ==============
@dataclass
class PollerResult:
  socket: zmq.Socket
  identity: Optional[bytes]
  data: Any


def poll_sockets(
  sockets: List[zmq.Socket],
  timeout: int = -1,
) -> List[PollerResult]:
  """轮询多个 socket。"""
  poller = zmq.Poller()
  for sock in sockets:
    poller.register(sock, zmq.POLLIN)

  results: List[PollerResult] = []
  ready = dict(poller.poll(timeout))
  for sock in sockets:
    if sock in ready:
      result = recv_pyobj(sock, timeout=0)
      if result:
        identity, data = result
        results.append(PollerResult(sock, identity, data))
  return results
