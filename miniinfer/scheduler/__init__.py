"""Scheduler package exports."""

from .prefill_adder import AddReqResult, CLIP_MAX_NEW_TOKENS, PrefillAdder
from .scheduler_batch import (
  BatchResult,
  BatchType,
  ChunkedReq,
  ForwardBatch,
  ForwardMode,
  Req,
  ScheduledBatch,
  clamp_position,
  compute_positions_extend,
)

__all__ = [
  "AddReqResult",
  "BatchResult",
  "BatchType",
  "ChunkedReq",
  "CLIP_MAX_NEW_TOKENS",
  "ForwardBatch",
  "ForwardMode",
  "PrefillAdder",
  "Req",
  "ScheduledBatch",
  "Scheduler",
  "clamp_position",
  "compute_positions_extend",
]


def __getattr__(name: str):
  if name == "Scheduler":
    from .scheduler import Scheduler

    return Scheduler
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
