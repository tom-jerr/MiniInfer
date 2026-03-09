import torch

from miniinfer.layers.sample import BatchSamplingArgs, Sampler
from miniinfer.scheduler.scheduler_batch import ForwardBatch, ForwardMode, Req, ScheduledBatch
from miniinfer.utils.sampling_params import SamplingParams


def test_forward_batch_sampling_metadata_all_greedy_skips_tensors():
  reqs = [
    Req([1, 2, 3], sampling_params=SamplingParams(temperature=0.0, max_tokens=1)),
    Req([4, 5], sampling_params=SamplingParams(temperature=0.0, max_tokens=1)),
  ]
  batch = ScheduledBatch(
    reqs=reqs,
    forward_mode=ForwardMode.DECODE,
    device=torch.device("cuda"),
    input_ids=torch.tensor([[3], [5]], dtype=torch.long),
    seq_lens=torch.tensor([3, 2], dtype=torch.int64),
    seq_lens_cpu=torch.tensor([3, 2], dtype=torch.int64),
  )

  fb = ForwardBatch.init_new(batch)
  assert fb.sampling_is_all_greedy is True
  assert fb.sampling_temperatures is None
  assert fb.sampling_top_ps is None
  assert fb.sampling_top_ks is None


def test_forward_batch_sampling_metadata_optional_tensors():
  reqs = [
    Req([1], sampling_params=SamplingParams(temperature=0.0, max_tokens=1, top_k=0, top_p=1.0)),
    Req([2], sampling_params=SamplingParams(temperature=0.7, max_tokens=1, top_k=5, top_p=1.0)),
    Req([3], sampling_params=SamplingParams(temperature=0.7, max_tokens=1, top_k=0, top_p=1.0)),
  ]
  batch = ScheduledBatch(
    reqs=reqs,
    forward_mode=ForwardMode.DECODE,
    device=torch.device("cuda"),
    input_ids=torch.tensor([[1], [2], [3]], dtype=torch.long),
    seq_lens=torch.tensor([1, 1, 1], dtype=torch.int64),
    seq_lens_cpu=torch.tensor([1, 1, 1], dtype=torch.int64),
  )

  fb = ForwardBatch.init_new(batch)
  assert fb.sampling_is_all_greedy is False
  assert isinstance(fb.sampling_temperatures, torch.Tensor)
  assert isinstance(fb.sampling_top_ks, torch.Tensor)
  assert fb.sampling_top_ps is None
  assert fb.sampling_max_top_k == 5


def test_sampler_all_greedy_argmax_path_cpu():
  sampler = Sampler(device="cpu", vocab_size=3, prefer_flashinfer=False)
  logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
  out = sampler.sample(logits, BatchSamplingArgs(temperatures=None))
  assert out.tolist() == [2, 0]


def test_sampler_topk_k1_is_deterministic_cpu_fallback():
  sampler = Sampler(device="cpu", vocab_size=3, prefer_flashinfer=False)
  logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=torch.float32)
  temps = torch.tensor([1.0, 1.0], dtype=torch.float32)
  top_ks = torch.tensor([1, 1], dtype=torch.int32)
  args = BatchSamplingArgs(temperatures=temps, top_k=top_ks, top_p=None, max_top_k=1)
  out = sampler.sample(logits, args)
  assert out.tolist() == [2, 0]
