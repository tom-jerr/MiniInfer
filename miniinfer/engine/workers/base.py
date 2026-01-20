"""Worker 进程基类和实现"""

from abc import ABC, abstractmethod
import multiprocessing as mp
import signal
import logging
import os
from typing import Optional, Any, Dict
from enum import Enum, auto
import zmq

from engine.ipc.zmq_channel import ZMQChannel, SocketType
from engine.ipc.protocol import MessageType, Request, Response

logger = logging.getLogger(__name__)


class WorkerState(Enum):
    """Worker 状态"""

    INIT = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class BaseWorker(ABC):
    """
    Worker 基类

    所有 Worker 进程都继承自此类，实现统一的生命周期管理和 IPC
    """

    def __init__(
        self,
        name: str,
        zmq_context: Optional[zmq.Context] = None,
    ):
        self.name = name
        self.state = WorkerState.INIT
        self.zmq_context = zmq_context or zmq.Context.instance()
        self._shutdown_event = mp.Event()

        # 设置信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"{self.name} received signal {signum}, shutting down...")
        self._shutdown_event.set()

    @property
    def should_stop(self) -> bool:
        return self._shutdown_event.is_set()

    @abstractmethod
    def setup(self):
        """初始化 Worker，设置 ZMQ 通道等"""
        pass

    @abstractmethod
    def process(self) -> bool:
        """
        处理一轮消息

        Returns:
            是否继续运行
        """
        pass

    @abstractmethod
    def cleanup(self):
        """清理资源"""
        pass

    def run(self):
        """主循环"""
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
                    logger.error(f"{self.name} error in process: {e}")
                    self.state = WorkerState.ERROR
                    break

            self.state = WorkerState.STOPPING
            logger.info(f"{self.name} stopping...")

        finally:
            self.cleanup()
            self.state = WorkerState.STOPPED
            logger.info(f"{self.name} stopped")

    def stop(self):
        """请求停止"""
        self._shutdown_event.set()


def worker_process_entry(worker_class, *args, **kwargs):
    """
    Worker 进程入口函数

    用于 multiprocessing.Process target
    """
    worker = worker_class(*args, **kwargs)
    worker.run()
