"""
Scheduler - 单机版调度器

负责:
1. 管理请求队列 (waiting, running, finished)
2. 组织 batch (prefill-only / decode-only / mixed)
3. 调用 KVCacheManager 分配 KV Cache
4. 增量流式解码
5. Token 预算控制，防止 OOM
6. 支持 Chunked Prefill
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import torch

from miniinfer.config.engine.config import EngineConfig
from miniinfer.kvcache.kv_cache_manager import KVCacheManager
from miniinfer.kvcache.memory_budget import PrefillAdder, TokenBudgetAdder
from miniinfer.engine.scheduler_batch import (
    Req,
    ChunkedReq,
    ScheduledBatch,
    ForwardBatch,
    ForwardMode,
    BatchResult,
)

logger = logging.getLogger(__name__)


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
        self.max_num_batched_tokens = getattr(
            config, "max_num_batched_tokens", config.max_total_tokens
        )

        # ============ Token Budget 控制 ============
        self.enable_chunked_prefill = getattr(config, "enable_chunked_prefill", False)
        self.chunked_prefill_size = getattr(config, "chunked_prefill_size", 4096)

        # 请求队列
        self.waiting_queue: List[Req] = []
        self.running_batch: ScheduledBatch = ScheduledBatch(reqs=[])
        self.cur_batch: Optional[ScheduledBatch] = None
        self.last_batch: Optional[ScheduledBatch] = None
        self.num_retracted_reqs: int = 0

        # Chunked prefill 相关：追踪正在进行 chunked prefill 的请求
        self.chunking_reqs: List[ChunkedReq] = []

        # 已完成的请求
        self.finished_reqs: List[Req] = []

        # EOS token id
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

        logger.info(
            f"Scheduler initialized: max_batch_size={self.max_batch_size}, "
            f"max_num_batched_tokens={self.max_num_batched_tokens}, "
            f"enable_chunked_prefill={self.enable_chunked_prefill}"
        )

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

        核心策略（类似 SGLang PrefillAdder）:
        1. 优先保证 decode 请求：reserved_size = running_bs
        2. prefill_budget = max_num_batched_tokens - running_bs
        3. 先添加所有 decode 请求，再添加 prefill 请求
        4. 如果 prefill 请求太大，启用 chunked prefill
        5. Chunked 请求不会进入 decode 阶段，直到完成所有 chunk

        Returns:
            ScheduledBatch: 调度的批次（可能是 MIXED、EXTEND 或 DECODE）
        """
        # ========== Step 0: 过滤已完成的请求 ==========
        self._filter_batch(self.running_batch)
        running_bs = len(self.running_batch.reqs)

        # ========== Step 1: 创建 PrefillAdder ==========
        # reserved_size = running_bs（为 decode 预留的 token 数）
        # prefill_budget = max_num_batched_tokens - running_bs
        prefill_budget = self.max_num_batched_tokens - running_bs

        prefill_adder = PrefillAdder(
            prefill_budget=prefill_budget,
            reserved_size=running_bs,
            kv_cache_mgr=self.kv_cache_mgr,
            max_batch_size=self.max_batch_size,
        )

        # ========== Step 2: 添加 decode 请求 ==========
        # Decode 请求优先级最高，每个请求消耗 1 token
        decode_reqs = []
        for req in self.running_batch.reqs:
            if prefill_adder.add_decode(req):
                decode_reqs.append(req)

        # ========== Step 3: 处理 chunking 中的请求（继续之前的 chunk）==========
        # 这些请求已经开始 prefill 但还没完成
        continuing_chunks = []
        remaining_chunks = []

        for chunked_req in self.chunking_reqs:
            req = chunked_req.req
            # 更新 cached_len（之前 chunk 已经处理的）
            cached_len = chunked_req.cached_len + chunked_req.chunk_size
            remaining_len = req.total_input_len - cached_len

            if remaining_len <= 0:
                # 这个请求已经完成所有 chunk，可以进入 decode
                req.is_chunked = False
                # 将加入 running_batch（在下一轮 decode）
                self.running_batch.reqs.append(req)
                continue

            # 尝试继续这个 chunked 请求
            chunk_size = min(remaining_len, prefill_adder.available_prefill_budget)
            if chunk_size > 0:
                new_chunked = ChunkedReq(req, cached_len, chunk_size)
                continuing_chunks.append(new_chunked)
                prefill_adder.current_prefill_tokens += chunk_size
            else:
                # 预算不足，保留到下一轮
                remaining_chunks.append(ChunkedReq(req, cached_len, 0))

        self.chunking_reqs = remaining_chunks

        # ========== Step 4: 添加新的 prefill 请求 ==========
        prefill_reqs = []
        new_chunked_reqs = []
        remaining_waiting = []

        for req in self.waiting_queue:
            # 计算 prefix 匹配
            self.kv_cache_mgr.prefix_for_waiting_req(req)
            extend_len = req.extend_input_len

            # 尝试添加 prefill
            result = prefill_adder.try_add_prefill(
                req,
                chunk_size=(
                    self.chunked_prefill_size if self.enable_chunked_prefill else None
                ),
            )

            if result is None:
                # 无法添加（预算不足或 KV cache 满）
                remaining_waiting.append(req)
                continue

            req_added, actual_len, is_chunked = result

            if is_chunked:
                # 创建 ChunkedReq
                chunked_req = ChunkedReq(
                    req=req,
                    cached_len=req.cache_protected_len,
                    chunk_size=actual_len,
                )
                new_chunked_reqs.append(chunked_req)
            else:
                # 完整 prefill
                prefill_reqs.append(req)

        # 更新 waiting queue
        self.waiting_queue = remaining_waiting

        # ========== Step 5: 构建调度批次 ==========
        # 合并所有需要 prefill 的请求
        all_extend_reqs = (
            prefill_reqs
            + [c.req for c in continuing_chunks]
            + [c.req for c in new_chunked_reqs]
        )
        all_chunked_reqs = continuing_chunks + new_chunked_reqs

        # 更新 chunking_reqs（未完成的 chunk）
        for chunked_req in all_chunked_reqs:
            if not chunked_req.is_last_chunk:
                self.chunking_reqs.append(chunked_req)

        # 根据情况返回不同类型的 batch
        if len(all_extend_reqs) > 0 and len(decode_reqs) > 0:
            # MIXED batch: 同时处理 extend 和 decode
            return self._build_mixed_batch(
                extend_reqs=all_extend_reqs,
                decode_reqs=decode_reqs,
                chunked_reqs=all_chunked_reqs,
                device=device,
            )
        elif len(all_extend_reqs) > 0:
            # EXTEND only batch
            return self._build_extend_batch(
                extend_reqs=all_extend_reqs,
                chunked_reqs=all_chunked_reqs,
                device=device,
            )
        elif len(decode_reqs) > 0:
            # DECODE only batch
            return self._build_decode_batch(device=device)
        else:
            # 没有请求
            return None

    def _build_extend_batch(
        self,
        extend_reqs: List[Req],
        chunked_reqs: List[ChunkedReq],
        device: torch.device,
    ) -> ScheduledBatch:
        """构建 EXTEND only 批次"""
        batch = ScheduledBatch.init_new(extend_reqs, device=device)
        batch.forward_mode = ForwardMode.EXTEND

        # 处理 chunked 请求的特殊 input_ids
        self._prepare_chunked_input_ids(batch, chunked_reqs)

        self.kv_cache_mgr.prepare_for_extend(batch)

        # 非 chunked 的请求完成 prefill 后加入 running_batch
        for req in extend_reqs:
            if not req.is_chunked:
                self.running_batch.reqs.append(req)
                # 更新 running_batch 元数据
                self._update_running_batch_metadata(batch, req)

        return batch

    def _build_decode_batch(self, device: torch.device) -> ScheduledBatch:
        """构建 DECODE only 批次"""
        if len(self.running_batch.reqs) == 0:
            return None

        self.kv_cache_mgr.prepare_for_decode(self.running_batch)
        self.running_batch.forward_mode = ForwardMode.DECODE
        return self.running_batch

    def _build_mixed_batch(
        self,
        extend_reqs: List[Req],
        decode_reqs: List[Req],
        chunked_reqs: List[ChunkedReq],
        device: torch.device,
    ) -> ScheduledBatch:
        """
        构建 MIXED 批次（同时处理 extend 和 decode）

        这是 chunked prefill 的核心：在一个 batch 中同时处理
        decode 请求和 prefill 请求
        """
        # 创建包含所有请求的 batch
        all_reqs = extend_reqs + decode_reqs
        batch = ScheduledBatch.init_new(all_reqs, device=device)
        batch.forward_mode = ForwardMode.MIXED
        batch.decoding_reqs = decode_reqs

        # 处理 chunked 请求的特殊 input_ids
        self._prepare_chunked_input_ids(batch, chunked_reqs)

        # 分别准备 extend 和 decode 的 KV cache
        # 注意：这里需要 KVCacheManager 支持 mixed batch
        self.kv_cache_mgr.prepare_for_extend(batch)

        # 非 chunked 的 extend 请求完成后加入 running_batch
        for req in extend_reqs:
            if not req.is_chunked and req not in self.running_batch.reqs:
                self.running_batch.reqs.append(req)
                self._update_running_batch_metadata(batch, req)

        return batch

    def _prepare_chunked_input_ids(
        self,
        batch: ScheduledBatch,
        chunked_reqs: List[ChunkedReq],
    ):
        """
        为 chunked 请求准备特殊的 input_ids

        Chunked 请求只处理部分 token，需要调整 input_ids
        """
        # TODO: 实现 chunked input_ids 的构建
        # 目前假设 batch.input_ids 已经正确设置
        pass

    def _update_running_batch_metadata(self, batch: ScheduledBatch, req: Req):
        """更新 running_batch 的元数据

        当请求从 extend batch 移动到 running_batch 时，需要同步元数据。
        """
        # 获取请求在 extend batch 中的索引
        try:
            req_idx_in_batch = batch.reqs.index(req)
        except ValueError:
            return

        # 获取该请求的元数据
        if batch.req_pool_indices is None or batch.seq_lens is None:
            return

        req_pool_idx = batch.req_pool_indices[req_idx_in_batch : req_idx_in_batch + 1]
        seq_len = batch.seq_lens[req_idx_in_batch : req_idx_in_batch + 1]
        seq_len_cpu = (
            batch.seq_lens_cpu[req_idx_in_batch : req_idx_in_batch + 1]
            if batch.seq_lens_cpu is not None
            else None
        )

        # 更新 running_batch 的元数据
        if self.running_batch.req_pool_indices is None:
            self.running_batch.req_pool_indices = req_pool_idx
            self.running_batch.seq_lens = seq_len
            self.running_batch.seq_lens_cpu = seq_len_cpu
        else:
            self.running_batch.req_pool_indices = torch.cat(
                [self.running_batch.req_pool_indices, req_pool_idx]
            )
            self.running_batch.seq_lens = torch.cat(
                [self.running_batch.seq_lens, seq_len]
            )
            if seq_len_cpu is not None and self.running_batch.seq_lens_cpu is not None:
                self.running_batch.seq_lens_cpu = torch.cat(
                    [self.running_batch.seq_lens_cpu, seq_len_cpu]
                )

    def get_budget_stats(self) -> dict:
        """获取当前 token 预算统计信息"""
        running_bs = len(self.running_batch.reqs)
        return {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "running_bs": running_bs,
            "prefill_budget": self.max_num_batched_tokens - running_bs,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "chunking_reqs": len(self.chunking_reqs),
        }

    def _filter_batch(self, batch: ScheduledBatch):
        """过滤掉已完成的请求，并同步更新相关元数据"""
        if not batch.reqs:
            return

        original_len = len(batch.reqs)

        # 找出未完成请求的索引
        keep_indices = []
        kept_reqs = []
        for i, req in enumerate(batch.reqs):
            if not req.finished:
                keep_indices.append(i)
                kept_reqs.append(req)

        batch.reqs = kept_reqs

        # 如果没有请求被过滤，直接返回
        if len(keep_indices) == original_len:
            return

        # 同步更新元数据
        if len(keep_indices) > 0 and batch.req_pool_indices is not None:
            keep_indices_tensor = torch.tensor(
                keep_indices, device=batch.req_pool_indices.device
            )
            batch.req_pool_indices = batch.req_pool_indices[keep_indices_tensor]
            batch.seq_lens = batch.seq_lens[keep_indices_tensor]
            if batch.seq_lens_cpu is not None:
                batch.seq_lens_cpu = batch.seq_lens_cpu[keep_indices]
        elif len(keep_indices) == 0:
            batch.req_pool_indices = None
            batch.seq_lens = None
            batch.seq_lens_cpu = None

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

    def get_request_output(self, req_id: int) -> Optional[str]:
        """
        获取指定请求的完整输出文本

        Args:
            req_id: 请求 ID

        Returns:
            完整的输出文本，如果请求不存在则返回 None
        """
        return self.incremental_decoder.get_full_text(req_id)
