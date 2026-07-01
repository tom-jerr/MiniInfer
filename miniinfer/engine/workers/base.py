"""Worker 进程基类。

所有 worker 进程继承自 `BaseWorker`，实现统一的生命周期（setup → run loop →
cleanup）与 ZMQ 通道持有。shutdown 通过 `multiprocessing.Event` + 信号触发。
"""
from __future__ import annotations

import multiprocessing as mp
import signal
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Optional

import zmq

from miniinfer.utils import get_logger

logger = get_logger(__name__)


class WorkerState(Enum):
  """Worker 状态。"""

  INIT = auto()
  RUNNING = auto()
  STOPPING = auto()
  STOPPED = auto()
  ERROR = auto()


class BaseWorker(ABC):
  """Worker 基类。"""

  def __init__(self, name: str, zmq_context: Optional[zmq.Context] = None):
    self.name = name
    self.state = WorkerState.INIT
    # 每个 worker 进程独立 context（spawn 后 zmq.Context.instance() 是新的）。
    self.zmq_context = zmq_context or zmq.Context.instance()
    self._shutdown_event = mp.Event()

    # 子进程里注册信号处理（仅当前进程）。
    signal.signal(signal.SIGTERM, self._signal_handler)
    signal.signal(signal.SIGINT, self._signal_handler)

  def _signal_handler(self, signum, frame):
    logger.info(f"{self.name} received signal {signum}, shutting down...")
    self._shutdown_event.set()

  @property
  def should_stop(self) -> bool:
    return self._shutdown_event.is_set()

  @abstractmethod
  def setup(self) -> None:
    """初始化 Worker：加载资源、建立 ZMQ 通道等。"""
    ...

  @abstractmethod
  def process(self) -> bool:
    """处理一轮消息。返回是否继续运行。"""
    ...

  @abstractmethod
  def cleanup(self) -> None:
    """清理资源。"""
    ...

  def run(self) -> None:
    """主循环。"""
    try:
      logger.info(f"{self.name} starting...")
      self.setup()
      self.state = WorkerState.RUNNING
      logger.info(f"{self.name} running")

      while not self.should_stop:
        try:
          if not self.process():
            break
        except Exception as e:
          logger.exception(f"{self.name} error in process: {e}")
          self.state = WorkerState.ERROR
          break

      self.state = WorkerState.STOPPING
      logger.info(f"{self.name} stopping...")
    finally:
      try:
        self.cleanup()
      except Exception as e:
        logger.exception(f"{self.name} error in cleanup: {e}")
      self.state = WorkerState.STOPPED
      logger.info(f"{self.name} stopped")

  def stop(self) -> None:
    """请求停止。"""
    self._shutdown_event.set()


def worker_process_entry(worker_class, *args, **kwargs):
  """Worker 进程入口函数，用于 multiprocessing.Process target。"""
  worker = worker_class(*args, **kwargs)
  worker.run()
