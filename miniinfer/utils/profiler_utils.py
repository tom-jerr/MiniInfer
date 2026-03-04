"""Utilities for torch.profiler stage annotations."""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager, nullcontext
from typing import Any, Callable

try:
    from torch.profiler import record_function
except Exception:  # pragma: no cover - fallback for environments without torch.profiler
    record_function = None


def _is_stage_profile_enabled() -> bool:
    value = os.getenv("MINIINFER_PROFILE_STAGES", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _record_scope(name: str):
    if record_function is None or not _is_stage_profile_enabled():
        return nullcontext()
    return record_function(name)


@contextmanager
def stage(name: str):
    """Create a named stage range visible in torch profiler traces."""
    with _record_scope(name):
        yield


def _wrap_with_stage(func: Callable[..., Any], stage_name: str):
    if getattr(func, "__miniinfer_profiled__", False):
        return func

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with stage(stage_name):
            return func(*args, **kwargs)

    wrapped.__miniinfer_profiled__ = True
    return wrapped


def profile_methods(
    component: str,
    include_private: bool = True,
    include_dunder: bool = False,
):
    """
    Class decorator that wraps methods with torch.profiler record_function ranges.

    Stage names follow: ``stage::<component>.<method_name>``.
    """

    def decorate(cls):
        for name, attr in list(vars(cls).items()):
            if name.startswith("__") and name.endswith("__"):
                if not include_dunder and name not in {"__init__", "__enter__", "__exit__"}:
                    continue
            elif name.startswith("_") and not include_private:
                continue

            stage_name = f"stage::{component}.{name}"

            if isinstance(attr, staticmethod):
                wrapped = _wrap_with_stage(attr.__func__, stage_name)
                setattr(cls, name, staticmethod(wrapped))
            elif isinstance(attr, classmethod):
                wrapped = _wrap_with_stage(attr.__func__, stage_name)
                setattr(cls, name, classmethod(wrapped))
            elif callable(attr):
                setattr(cls, name, _wrap_with_stage(attr, stage_name))

        return cls

    return decorate
