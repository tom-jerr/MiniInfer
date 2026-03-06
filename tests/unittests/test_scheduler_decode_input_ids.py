import types

import torch

from miniinfer.scheduler.scheduler import Scheduler
from miniinfer.scheduler.scheduler_batch import Req, ScheduledBatch
from miniinfer.utils.sampling_params import SamplingParams


class _FakeTokenizer:
  eos_token_id = 0


class _FakeKVCacheManager:
  def available_tokens(self) -> int:
    return 1_000_000

  def prepare_for_decode(self, batch: ScheduledBatch, skip_input_ids: bool = False):
    from miniinfer.scheduler.scheduler_batch import ForwardMode

    batch.forward_mode = ForwardMode.DECODE
    bs = len(batch.reqs)
    if bs == 0:
      return
    if not skip_input_ids:
      last_tokens = [int(req.output_ids[-1]) for req in batch.reqs]
      batch.input_ids = torch.tensor(last_tokens, dtype=torch.int64, device="cpu")
    else:
      # sglang 风格：output_ids 存储了 -future_indices
      if batch.output_ids is not None and batch.output_ids.shape[0] == bs:
        batch.input_ids = batch.output_ids
      else:
        last_tokens = [int(req.output_ids[-1]) for req in batch.reqs]
        batch.input_ids = torch.tensor(last_tokens, dtype=torch.int64, device="cpu")
    batch.output_ids = None


def _make_scheduler() -> Scheduler:
  cfg = types.SimpleNamespace(
    max_num_seqs=8,
    max_extend_len=128,
    page_size=16,
    gpu_memory_utilization=0.6,
    enable_chunked_prefill=False,
    chunked_prefill_size=64,
    enable_overlap=True,
  )
  return Scheduler(cfg, _FakeTokenizer(), _FakeKVCacheManager())


def test_decode_builds_input_ids_when_not_skipped():
  scheduler = _make_scheduler()

  req = Req([1, 2, 3], SamplingParams(max_tokens=8))
  req.output_ids = [42]

  batch = ScheduledBatch.init_new([req], device="cpu")
  batch.req_pool_indices = torch.tensor([0], dtype=torch.int64, device="cpu")
  seq_len = len(req.origin_input_ids) + len(req.output_ids)
  batch.seq_lens = torch.tensor([seq_len], dtype=torch.int64, device="cpu")
  batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)

  scheduler.running_batch = batch

  out = scheduler._get_decode_batch(device="cpu", skip_decode_input_ids=False)
  assert out is not None
  assert out.input_ids is not None


def test_decode_can_skip_input_ids_for_placeholder_path():
  scheduler = _make_scheduler()

  req = Req([1, 2, 3], SamplingParams(max_tokens=8))
  req.output_ids = [42]

  batch = ScheduledBatch.init_new([req], device="cpu")
  batch.req_pool_indices = torch.tensor([0], dtype=torch.int64, device="cpu")
  seq_len = len(req.origin_input_ids) + len(req.output_ids)
  batch.seq_lens = torch.tensor([seq_len], dtype=torch.int64, device="cpu")
  batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)

  scheduler.running_batch = batch

  out = scheduler._get_decode_batch(device="cpu", skip_decode_input_ids=True)
  assert out is not None
  # sglang 风格：没有 output_ids 时 fallback 到同步构建 input_ids
  assert out.input_ids is not None
  assert out.input_ids.tolist() == [42]


def test_decode_uses_output_ids_as_placeholder():
  """sglang 风格：output_ids 中的 future indices 被传递为 input_ids"""
  scheduler = _make_scheduler()

  req = Req([1, 2, 3], SamplingParams(max_tokens=8))
  req.output_ids = [42]

  batch = ScheduledBatch.init_new([req], device="cpu")
  batch.req_pool_indices = torch.tensor([0], dtype=torch.int64, device="cpu")
  seq_len = len(req.origin_input_ids) + len(req.output_ids)
  batch.seq_lens = torch.tensor([seq_len], dtype=torch.int64, device="cpu")
  batch.seq_lens_cpu = torch.tensor([seq_len], dtype=torch.int64)
  # 模拟 run_batch_async 存储的 -future_indices
  batch.output_ids = torch.tensor([-5], dtype=torch.int64, device="cpu")

  scheduler.running_batch = batch

  out = scheduler._get_decode_batch(device="cpu", skip_decode_input_ids=True)
  assert out is not None
  assert out.input_ids is not None
  # 应该使用 output_ids（负数占位符），而非同步构建的真实 token
  assert out.input_ids.tolist() == [-5]


def test_decode_filter_output_ids_on_batch_shrink():
  """当请求完成被过滤时，output_ids 也同步过滤"""
  scheduler = _make_scheduler()

  req1 = Req([1, 2, 3], SamplingParams(max_tokens=8))
  req1.output_ids = [42]
  req2 = Req([4, 5, 6], SamplingParams(max_tokens=8))
  req2.output_ids = [99]
  req2.finished = True  # 标记已完成

  batch = ScheduledBatch.init_new([req1, req2], device="cpu")
  batch.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device="cpu")
  batch.seq_lens = torch.tensor([4, 4], dtype=torch.int64, device="cpu")
  batch.seq_lens_cpu = torch.tensor([4, 4], dtype=torch.int64)
  # 模拟两个请求的 future indices
  batch.output_ids = torch.tensor([-10, -11], dtype=torch.int64, device="cpu")

  scheduler.running_batch = batch

  # filter_batch 应该移除 req2 并同步过滤 output_ids
  scheduler._filter_batch(batch)
  assert len(batch.reqs) == 1
  assert batch.output_ids is not None
  assert batch.output_ids.tolist() == [-10]
