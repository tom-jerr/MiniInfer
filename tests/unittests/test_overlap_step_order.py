import types

from miniinfer.engine.llm_engine import LLMEngine, StepOutput
from miniinfer.scheduler.scheduler_batch import ScheduledBatch


class _FakeOverlapExecutor:
  def __init__(self):
    self._pending = True
    self.calls: list[str] = []

  def has_pending(self) -> bool:
    self.calls.append("has_pending")
    return self._pending

  def process_pending_batch(self, _process_func):
    self.calls.append("process_pending_batch")
    self._pending = False
    return StepOutput(outputs=[])

  def run_batch_async(self, *_args, **_kwargs):
    self.calls.append("run_batch_async")


class _FakeScheduler:
  def __init__(self, overlap: _FakeOverlapExecutor):
    self._overlap = overlap
    self.calls: list[str] = []
    self.pending_release_reqs = [object()]
    self.running_batch = ScheduledBatch(reqs=[])

  def schedule(self, _device=None, *, skip_decode_input_ids: bool = False):
    self.calls.append("schedule")
    # The scheduler must not be called while a previous batch is still pending,
    # otherwise req.output_ids may be empty/stale and decode preparation can crash.
    assert not self._overlap._pending
    return None

  def drain_pending_releases(self):
    self.calls.append("drain_pending_releases")
    self.pending_release_reqs = []


def test_step_overlap_processes_pending_before_scheduling():
  engine = object.__new__(LLMEngine)
  overlap = _FakeOverlapExecutor()
  engine.overlap_executor = overlap
  engine.scheduler = _FakeScheduler(overlap)
  engine.config = types.SimpleNamespace(enable_overlap=True)
  engine.model_runner = types.SimpleNamespace(device=None, attn_backend=None)
  engine._last_batch = object()

  out = LLMEngine.step_overlap(engine)

  assert isinstance(out, StepOutput)
  assert "process_pending_batch" in overlap.calls
  assert "schedule" in engine.scheduler.calls
  # Verify ordering: process_pending_batch must come before schedule
  proc_idx = overlap.calls.index("process_pending_batch")
  sched_idx = engine.scheduler.calls.index("schedule")
  # Both are in separate call lists, but process_pending_batch was called
  # while pending was True, and schedule asserts pending is False, so
  # the ordering is correct if both succeed without assertion errors.
