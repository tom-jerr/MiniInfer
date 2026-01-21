# """
# ProcessController - 统一管理所有 Worker 进程

# 职责:
# - 启动/停止所有 Worker 进程
# - 监控进程健康状态
# - 处理信号 (graceful shutdown)
# - 进程异常重启
# """

# import multiprocessing as mp
# import signal
# import logging
# import time
# import os
# import zmq
# from typing import Dict, Optional, List, Callable, Any
# from dataclasses import dataclass, field
# from enum import Enum, auto

# from .workers.base import worker_process_entry, WorkerState
# from .workers.tokenizer_worker import TokenizerWorker
# from .workers.scheduler_worker import SchedulerWorker
# from .workers.detokenizer_worker import DetokenizerWorker

# logger = logging.getLogger(__name__)


# @dataclass
# class ProcessInfo:
#     """进程信息"""

#     name: str
#     process: Optional[mp.Process] = None
#     worker_class: Any = None
#     worker_args: tuple = field(default_factory=tuple)
#     worker_kwargs: dict = field(default_factory=dict)
#     restart_count: int = 0
#     max_restarts: int = 3
#     last_health_check: float = 0


# class ProcessController:
#     """
#     进程控制器

#     管理 Tokenizer, Scheduler, Detokenizer 三个进程的生命周期

#     Usage:
#         controller = ProcessController(model_path="...")
#         controller.start_all()

#         # ... 运行中 ...

#         controller.stop_all()
#     """

#     # IPC 地址常量
#     IPC_BASE_DIR = "/tmp/miniinfer"

#     def __init__(
#         self,
#         model_path: str,
#         # KV Cache 配置
#         max_tokens: int = 65536,
#         max_requests: int = 256,
#         max_context_len: int = 4096,
#         num_layers: int = 32,
#         num_heads: int = 32,
#         head_dim: int = 128,
#         # Scheduler 配置
#         max_batch_size: int = 32,
#         max_prefill_tokens: int = 4096,
#         enable_prefix_cache: bool = True,
#         # 其他配置
#         health_check_interval: float = 5.0,
#         auto_restart: bool = True,
#     ):
#         self.model_path = model_path
#         self.health_check_interval = health_check_interval
#         self.auto_restart = auto_restart

#         # 保存配置
#         self.config = {
#             "max_tokens": max_tokens,
#             "max_requests": max_requests,
#             "max_context_len": max_context_len,
#             "num_layers": num_layers,
#             "num_heads": num_heads,
#             "head_dim": head_dim,
#             "max_batch_size": max_batch_size,
#             "max_prefill_tokens": max_prefill_tokens,
#             "enable_prefix_cache": enable_prefix_cache,
#         }

#         # 生成 IPC 地址
#         self.ipc_addresses = self._generate_ipc_addresses()

#         # 进程信息
#         self.processes: Dict[str, ProcessInfo] = {}

#         # ZMQ Context (主进程)
#         self.zmq_context: Optional[zmq.Context] = None

#         # 控制标志
#         self._running = False
#         self._shutdown_event = mp.Event()

#         # 设置信号处理
#         signal.signal(signal.SIGTERM, self._signal_handler)
#         signal.signal(signal.SIGINT, self._signal_handler)

#     def _generate_ipc_addresses(self) -> Dict[str, str]:
#         """生成 IPC 地址"""
#         pid = os.getpid()
#         base = f"{self.IPC_BASE_DIR}/{pid}"
#         os.makedirs(base, exist_ok=True)

#         return {
#             # 客户端 -> Tokenizer
#             "client_to_tokenizer": f"ipc://{base}/client_tokenizer.sock",
#             # Tokenizer -> Scheduler
#             "tokenizer_to_scheduler": f"ipc://{base}/tokenizer_scheduler.sock",
#             # Scheduler -> Detokenizer
#             "scheduler_to_detokenizer": f"ipc://{base}/scheduler_detokenizer.sock",
#             # Detokenizer -> 客户端
#             "detokenizer_to_client": f"ipc://{base}/detokenizer_client.sock",
#         }

#     def _signal_handler(self, signum, frame):
#         """信号处理"""
#         logger.info(f"Received signal {signum}, initiating shutdown...")
#         self._shutdown_event.set()

#     def _create_process_infos(self):
#         """创建进程信息"""
#         # Tokenizer Worker
#         self.processes["tokenizer"] = ProcessInfo(
#             name="tokenizer",
#             worker_class=TokenizerWorker,
#             worker_kwargs={
#                 "model_path": self.model_path,
#                 "client_address": self.ipc_addresses["client_to_tokenizer"],
#                 "scheduler_address": self.ipc_addresses["tokenizer_to_scheduler"],
#             },
#         )

#         # Scheduler Worker
#         self.processes["scheduler"] = ProcessInfo(
#             name="scheduler",
#             worker_class=SchedulerWorker,
#             worker_kwargs={
#                 "max_tokens": self.config["max_tokens"],
#                 "max_requests": self.config["max_requests"],
#                 "max_context_len": self.config["max_context_len"],
#                 "num_layers": self.config["num_layers"],
#                 "num_heads": self.config["num_heads"],
#                 "head_dim": self.config["head_dim"],
#                 "max_batch_size": self.config["max_batch_size"],
#                 "max_prefill_tokens": self.config["max_prefill_tokens"],
#                 "enable_prefix_cache": self.config["enable_prefix_cache"],
#                 "tokenizer_address": self.ipc_addresses["tokenizer_to_scheduler"],
#                 "detokenizer_address": self.ipc_addresses["scheduler_to_detokenizer"],
#             },
#         )

#         # Detokenizer Worker
#         self.processes["detokenizer"] = ProcessInfo(
#             name="detokenizer",
#             worker_class=DetokenizerWorker,
#             worker_kwargs={
#                 "model_path": self.model_path,
#                 "scheduler_address": self.ipc_addresses["scheduler_to_detokenizer"],
#                 "client_address": self.ipc_addresses["detokenizer_to_client"],
#             },
#         )

#     def _start_process(self, name: str) -> bool:
#         """启动单个进程"""
#         info = self.processes.get(name)
#         if info is None:
#             logger.error(f"Unknown process: {name}")
#             return False

#         if info.process is not None and info.process.is_alive():
#             logger.warning(f"Process {name} is already running")
#             return True

#         logger.info(f"Starting process: {name}")

#         ctx = mp.get_context("spawn")
#         process = ctx.Process(
#             target=worker_process_entry,
#             args=(info.worker_class,),
#             kwargs=info.worker_kwargs,
#             name=name,
#         )
#         process.start()

#         info.process = process
#         logger.info(f"Process {name} started with PID {process.pid}")

#         return True

#     def _stop_process(self, name: str, timeout: float = 5.0) -> bool:
#         """停止单个进程"""
#         info = self.processes.get(name)
#         if info is None or info.process is None:
#             return True

#         if not info.process.is_alive():
#             info.process = None
#             return True

#         logger.info(f"Stopping process: {name}")

#         # 发送 SIGTERM
#         info.process.terminate()
#         info.process.join(timeout=timeout)

#         if info.process.is_alive():
#             logger.warning(f"Process {name} did not stop gracefully, killing...")
#             info.process.kill()
#             info.process.join(timeout=1.0)

#         info.process = None
#         logger.info(f"Process {name} stopped")

#         return True

#     def start_all(self):
#         """启动所有进程"""
#         if self._running:
#             logger.warning("ProcessController is already running")
#             return

#         logger.info("Starting all worker processes...")

#         self.zmq_context = zmq.Context()
#         self._create_process_infos()

#         # 按顺序启动: Scheduler -> Tokenizer -> Detokenizer
#         # Scheduler 需要先启动，因为它 bind 地址
#         start_order = ["scheduler", "tokenizer", "detokenizer"]

#         for name in start_order:
#             if not self._start_process(name):
#                 logger.error(f"Failed to start {name}")
#                 self.stop_all()
#                 raise RuntimeError(f"Failed to start {name}")
#             time.sleep(0.5)  # 等待进程初始化

#         self._running = True
#         logger.info("All worker processes started")

#     def stop_all(self):
#         """停止所有进程"""
#         if not self._running:
#             return

#         logger.info("Stopping all worker processes...")

#         # 按相反顺序停止
#         stop_order = ["detokenizer", "tokenizer", "scheduler"]

#         for name in stop_order:
#             self._stop_process(name)

#         # 清理 ZMQ
#         if self.zmq_context:
#             self.zmq_context.term()
#             self.zmq_context = None

#         # 清理 IPC socket 文件
#         self._cleanup_ipc_files()

#         self._running = False
#         logger.info("All worker processes stopped")

#     def _cleanup_ipc_files(self):
#         """清理 IPC socket 文件"""
#         import shutil

#         pid = os.getpid()
#         base = f"{self.IPC_BASE_DIR}/{pid}"
#         if os.path.exists(base):
#             shutil.rmtree(base, ignore_errors=True)

#     def check_health(self) -> Dict[str, bool]:
#         """检查所有进程健康状态"""
#         health = {}
#         for name, info in self.processes.items():
#             if info.process is None:
#                 health[name] = False
#             else:
#                 health[name] = info.process.is_alive()
#         return health

#     def wait_for_shutdown(self):
#         """等待关闭信号"""
#         logger.info("Waiting for shutdown signal...")

#         while not self._shutdown_event.is_set():
#             # 检查进程健康
#             health = self.check_health()

#             for name, is_healthy in health.items():
#                 if not is_healthy:
#                     logger.warning(f"Process {name} is not healthy")

#                     if self.auto_restart:
#                         info = self.processes[name]
#                         if info.restart_count < info.max_restarts:
#                             logger.info(f"Restarting process {name}...")
#                             self._start_process(name)
#                             info.restart_count += 1
#                         else:
#                             logger.error(f"Process {name} exceeded max restarts")
#                             self._shutdown_event.set()
#                             break

#             time.sleep(self.health_check_interval)

#         self.stop_all()

#     def run(self):
#         """启动并运行直到收到关闭信号"""
#         try:
#             self.start_all()
#             self.wait_for_shutdown()
#         except Exception as e:
#             logger.error(f"Error running ProcessController: {e}")
#             raise
#         finally:
#             self.stop_all()

#     @property
#     def is_running(self) -> bool:
#         return self._running

#     def get_stats(self) -> Dict[str, Any]:
#         """获取运行状态"""
#         return {
#             "running": self._running,
#             "processes": {
#                 name: {
#                     "alive": info.process.is_alive() if info.process else False,
#                     "pid": info.process.pid if info.process else None,
#                     "restart_count": info.restart_count,
#                 }
#                 for name, info in self.processes.items()
#             },
#             "ipc_addresses": self.ipc_addresses,
#         }
