# """
# ZMQ 通信模块 - 直接使用 pyzmq

# 进程间通信拓扑:
#     Client ──DEALER──▶ ROUTER── Tokenizer ──PUSH──▶ PULL── Scheduler
#                                                               │
#     Client ◀──DEALER── ROUTER── Detokenizer ◀──PULL── PUSH───┘

# 同机器通信使用 Unix Domain Socket (ipc://), 跨机器使用 TCP (tcp://)
# """

# import zmq
# import os
# import pickle
# import logging
# from typing import Optional, Any, Tuple, List
# from dataclasses import dataclass

# logger = logging.getLogger(__name__)


# # ============== 地址生成工具 ==============


# def make_ipc_address(name: str, base_dir: str = "/tmp/miniinfer") -> str:
#     """
#     生成 IPC 地址 (Unix Domain Socket)

#     Args:
#         name: socket 名称
#         base_dir: socket 文件存放目录
#     """
#     os.makedirs(base_dir, exist_ok=True)
#     return f"ipc://{base_dir}/{name}.sock"


# def make_tcp_address(host: str = "127.0.0.1", port: int = 5555) -> str:
#     """生成 TCP 地址"""
#     return f"tcp://{host}:{port}"


# def cleanup_ipc_socket(address: str):
#     """清理 IPC socket 文件"""
#     if address.startswith("ipc://"):
#         sock_path = address[6:]
#         if os.path.exists(sock_path):
#             os.unlink(sock_path)


# # ============== 序列化工具 ==============


# def serialize(obj: Any) -> bytes:
#     """序列化对象"""
#     return pickle.dumps(obj)


# def deserialize(data: bytes) -> Any:
#     """反序列化对象"""
#     return pickle.loads(data)


# # ============== Socket 工厂函数 ==============


# def create_socket(
#     ctx: zmq.Context,
#     socket_type: int,
#     address: str,
#     bind: bool = False,
#     identity: Optional[str] = None,
# ) -> zmq.Socket:
#     """
#     创建并配置 ZMQ socket

#     Args:
#         ctx: ZMQ Context
#         socket_type: zmq.PUSH, zmq.PULL, zmq.ROUTER, zmq.DEALER 等
#         address: 地址 (ipc:// 或 tcp://)
#         bind: True=bind, False=connect
#         identity: socket 标识 (用于 ROUTER/DEALER)

#     Returns:
#         配置好的 socket
#     """
#     socket = ctx.socket(socket_type)

#     # 设置 identity
#     if identity:
#         socket.setsockopt_string(zmq.IDENTITY, identity)

#     # 设置 socket 选项
#     socket.setsockopt(zmq.LINGER, 0)  # 关闭时不等待

#     if bind:
#         cleanup_ipc_socket(address)
#         socket.bind(address)
#         logger.debug(f"Bound {socket_type} to {address}")
#     else:
#         socket.connect(address)
#         logger.debug(f"Connected {socket_type} to {address}")

#     return socket


# # ============== 发送/接收函数 ==============


# def send_pyobj(socket: zmq.Socket, obj: Any, identity: Optional[bytes] = None):
#     """
#     发送 Python 对象

#     Args:
#         socket: ZMQ socket
#         obj: 要发送的对象
#         identity: 目标 identity (仅 ROUTER socket 需要)
#     """
#     data = serialize(obj)

#     if identity is not None:
#         # ROUTER socket: [identity, '', data]
#         socket.send_multipart([identity, b"", data])
#     else:
#         socket.send(data)


# def recv_pyobj(
#     socket: zmq.Socket,
#     timeout: Optional[int] = None,
# ) -> Optional[Tuple[Optional[bytes], Any]]:
#     """
#     接收 Python 对象

#     Args:
#         socket: ZMQ socket
#         timeout: 超时时间 (毫秒), None=阻塞, 0=非阻塞

#     Returns:
#         (identity, obj) 元组, identity 仅 ROUTER socket 有效
#         超时返回 None
#     """
#     if timeout is not None:
#         if not socket.poll(timeout, zmq.POLLIN):
#             return None

#     # 检查是否是 ROUTER socket
#     socket_type = socket.getsockopt(zmq.TYPE)

#     if socket_type == zmq.ROUTER:
#         frames = socket.recv_multipart()
#         identity = frames[0]
#         data = frames[-1]
#     else:
#         identity = None
#         data = socket.recv()

#     obj = deserialize(data)
#     return identity, obj


# def recv_pyobj_nowait(socket: zmq.Socket) -> Optional[Tuple[Optional[bytes], Any]]:
#     """非阻塞接收"""
#     return recv_pyobj(socket, timeout=0)


# # ============== Poller 封装 ==============


# @dataclass
# class PollerResult:
#     """Poller 结果"""

#     socket: zmq.Socket
#     identity: Optional[bytes]
#     data: Any


# def poll_sockets(
#     sockets: List[zmq.Socket],
#     timeout: int = -1,
# ) -> List[PollerResult]:
#     """
#     轮询多个 socket

#     Args:
#         sockets: socket 列表
#         timeout: 超时时间 (毫秒), -1=阻塞

#     Returns:
#         有数据的 socket 及其数据列表
#     """
#     poller = zmq.Poller()
#     for sock in sockets:
#         poller.register(sock, zmq.POLLIN)

#     results = []
#     ready = dict(poller.poll(timeout))

#     for sock in sockets:
#         if sock in ready:
#             result = recv_pyobj(sock, timeout=0)
#             if result:
#                 identity, data = result
#                 results.append(PollerResult(sock, identity, data))

#     return results


# # ============== 高级封装: 双向通道 ==============


# class BiChannel:
#     """
#     双向通信通道

#     使用 PUSH/PULL 模式实现双向通信
#     """

#     def __init__(self, ctx: zmq.Context):
#         self.ctx = ctx
#         self.send_socket: Optional[zmq.Socket] = None
#         self.recv_socket: Optional[zmq.Socket] = None

#     def bind(self, send_addr: str, recv_addr: str):
#         """绑定发送和接收地址 (服务端)"""
#         self.send_socket = create_socket(self.ctx, zmq.PUSH, send_addr, bind=True)
#         self.recv_socket = create_socket(self.ctx, zmq.PULL, recv_addr, bind=True)

#     def connect(self, send_addr: str, recv_addr: str):
#         """连接发送和接收地址 (客户端)"""
#         self.send_socket = create_socket(self.ctx, zmq.PUSH, send_addr, bind=False)
#         self.recv_socket = create_socket(self.ctx, zmq.PULL, recv_addr, bind=False)

#     def send(self, obj: Any):
#         """发送对象"""
#         if self.send_socket:
#             send_pyobj(self.send_socket, obj)

#     def recv(self, timeout: Optional[int] = None) -> Optional[Any]:
#         """接收对象"""
#         if self.recv_socket:
#             result = recv_pyobj(self.recv_socket, timeout)
#             return result[1] if result else None
#         return None

#     def close(self):
#         """关闭通道"""
#         if self.send_socket:
#             self.send_socket.close()
#             self.send_socket = None
#         if self.recv_socket:
#             self.recv_socket.close()
#             self.recv_socket = None
