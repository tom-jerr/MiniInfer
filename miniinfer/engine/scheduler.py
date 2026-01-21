"""
Scheduler - 单机版调度器

负责:
1. 管理请求队列 (waiting, running, finished)
2. 组织 batch (prefill-only / decode-only / mixed)
3. 调用 KVCacheManager 分配 KV Cache
4. 增量流式解码
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import torch

from miniinfer.config.engine.config import EngineConfig
from miniinfer.kvcache.kv_cache_manager import KVCacheManager
from miniinfer.engine.scheduler_batch import (
    Req,
    ScheduledBatch,
    ForwardBatch,
    BatchResult,
)

class Scheduler:

    def __init__(
        self,
        config: EngineConfig,
        tokenizer: Any,
        kv_cache_mgr: KVCacheManager,
    ):
        """
        初始化调度器

        Args:
            config: 引擎配置
            tokenizer: 分词器
            kv_cache_mgr: KV Cache 管理器
        """
        self.engine_config = config
        self.tokenizer = tokenizer
        self.kv_cache_mgr = kv_cache_mgr
        self.max_batch_size = config.max_num_seqs
        self.max_num_batched_tokens = getattr(config, 'max_num_batched_tokens', config.max_total_tokens)

        # 请求队列
        self.waiting_queue: List[Req] = []
        self.running_batch: ScheduledBatch = ScheduledBatch(reqs=[])
        self.cur_batch: Optional[ScheduledBatch] = None
        self.last_batch: Optional[ScheduledBatch] = None
        self.num_retracted_reqs: int = 0

        # 已完成的请求
        self.finished_reqs: List[Req] = []

        # EOS token id
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)



    def step(self):
        batch = self.schedule()
        self.cur_batch = batch

        result = self.run_batch(batch)
        output_texts = self.process_batch_result(batch, result)
        return output_texts

    def add(self, req: Req):
        """添加新请求到等待队列"""
        self.waiting_queue.append(req)

    def has_unfinished(self) -> bool:
        """检查是否有未完成的请求"""
        return len(self.waiting_queue) > 0 or len(self.running_batch.reqs) > 0

    def get_num_unfinished(self) -> int:
        """获取未完成请求数量"""
        return len(self.waiting_queue) + len(self.running_batch.reqs)

    def schedule(self, device: torch.device) -> ScheduledBatch:
        """
        调度一个 batch

        策略:
        1. 优先处理 running queue 中的 decode 请求
        2. 然后从 waiting queue 添加 prefill 请求
        """
        # TODO(lzy): care about OOM
        running_bs = len(self.running_batch.reqs)
        can_run_list = []
        for req in self.waiting_queue:
            if running_bs >= self.max_batch_size:
                break
            self.kv_cache_mgr.prefix_for_waiting_req(req)
            can_run_list.append(req)
        self.waiting_queue = [
            req for req in self.waiting_queue if req not in can_run_list
        ]
        new_batch = ScheduledBatch.init_new(can_run_list, device=device)
        self.kv_cache_mgr.prepare_for_extend(new_batch)
        if new_batch is not None:
            new_batch.debug_metadata()
            return new_batch
        # decode batch process
        self._filter_batch(self.running_batch)
        self.kv_cache_mgr.prepare_for_decode(self.running_batch)
        self.running_batch.debug_metadata()
        return self.running_batch if self.running_batch is not None else None

    def _filter_batch(self, batch: ScheduledBatch):
        """过滤掉已完成的请求"""
        batch.reqs = [req for req in batch.reqs if req.finished is False]

    def run_batch(self, batch: ScheduledBatch) -> Any:
        forward_batch = ForwardBatch.init_new(batch)
        out = self.modelrunner.forward(forward_batch)
        logits_output = out.logits
        next_token_ids = self.modelrunner.sample(logits_output, forward_batch)

        return BatchResult(
            logits=logits_output,
            next_token_ids=next_token_ids,
        )

    def process_batch_result(
        self, batch: ScheduledBatch, result: BatchResult
    ) -> List[str]:
        """
        处理批次推理结果

        对每个请求进行增量解码，更新请求状态，处理完成的请求。

        Args:
            batch: 调度的批次
            result: 批次推理结果，包含 logits 和 next_token_ids
        """
        if batch is None or len(batch.reqs) == 0:
            return

        next_token_ids = result.next_token_ids

        # 确保 next_token_ids 在 CPU 上
        if isinstance(next_token_ids, torch.Tensor):
            next_token_ids = next_token_ids.cpu().tolist()

        output_texts = []
        # 处理每个请求
        finished_req_ids = []
        for i, req in enumerate(batch.reqs):
            token_id = next_token_ids[i]

            # 将新 token 添加到请求的输出
            req.output_ids.append(token_id)

            # 增量解码
            delta_text, is_finished = self.incremental_decoder.decode(
                req_id=req.req_id,
                token_id=token_id,
                eos_token_id=self.eos_token_id,
            )
            output_texts.append(delta_text)

            # 检查是否达到最大 token 数
            if len(req.output_ids) >= req.max_tokens:
                is_finished = True
                # 刷新剩余的 pending 文本
                remaining = self.incremental_decoder.flush(req.req_id)
                if remaining:
                    output_texts[-1] += remaining

            # 更新请求状态
            if is_finished:
                req.finished = True
                if token_id == self.eos_token_id:
                    req.finished_reason = "eos"
                else:
                    req.finished_reason = "max_tokens"
                finished_req_ids.append(req.req_id)

        # 处理完成的请求
        self._handle_finished_requests(batch, finished_req_ids)
        return output_texts

    def _handle_finished_requests(
        self, batch: ScheduledBatch, finished_req_ids: List[int]
    ):
        """
        处理已完成的请求

        将完成的请求从 running batch 移动到 finished 列表，
        并清理相关的 KV cache 和解码状态。
        """
        if not finished_req_ids:
            return

        finished_req_id_set = set(finished_req_ids)

        for req in batch.reqs:
            if req.req_id in finished_req_id_set:
                # 添加到完成列表
                self.finished_reqs.append(req)

                # 清理 KV cache
                self.kv_cache_mgr.release_request(req)

                # 清理解码状态
                self.incremental_decoder.cleanup(req.req_id)

    def get_request_output(self, req_id: int) -> Optional[str]:
        """
        获取指定请求的完整输出文本

        Args:
            req_id: 请求 ID

        Returns:
            完整的输出文本，如果请求不存在则返回 None
        """
        return self.incremental_decoder.get_full_text(req_id)
