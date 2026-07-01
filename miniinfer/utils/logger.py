"""vLLM-style structured logger for MiniInfer.

Output format mirrors vLLM V1 so logs are greppable and side-by-side
comparable with nano-vllm / vLLM::

    (EngineCore pid=12345) INFO  06-17 15:02:00 [model_runner.py:450]
        Starting to load model /root/models/Qwen3-0.6B...

The leading ``(Role pid=PID)`` prefix is added per-process so multi-process
setups (overlap workers / TP ranks) stay distinguishable in a single stream.

Color
-----

ANSI colors are applied when emitting to a TTY. Detection follows the
de-facto standard:

* **Disable** colors when stdout is not a TTY (pipe/file redirect), when
  ``NO_COLOR`` is set (https://no-color.org/), or when ``TERM=dumb``.
  Per the spec, ``NO_COLOR`` always wins — use it to override
  ``CLICOLOR_FORCE`` or ``MINIINFER_LOG_COLOR``.
* **Force enable** with ``CLICOLOR_FORCE=1`` or ``MINIINFER_LOG_COLOR=1``
  (the latter also accepts ``0`` to force-disable even on a TTY).

Usage::

    from miniinfer.utils import get_logger
    log = get_logger("engine")          # role = "EngineCore" inferred
    log.info("...")

Roles can also be set explicitly at process entry (e.g. workers)::

    set_process_role(f"Worker-{rank}")
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

_ROLE = "EngineCore"  # default; overlap/multi-process paths override via set_process_role
_CONFIGURED: set[str] = set()


# ---------------------------------------------------------------------------
# ANSI color support
# ---------------------------------------------------------------------------
# We use the bright/bold 9x palette — it reads well on the dark terminal
# backgrounds that are the common case for ML workflows.
_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_BLUE = "\033[94m"
_MAGENTA = "\033[95m"
_CYAN = "\033[96m"


# Per-level color map: (level prefix color, message color or "" for default).
_LEVEL_COLORS: dict[int, tuple[str, str]] = {
    logging.DEBUG:    (_CYAN,    _DIM),
    logging.INFO:     (_GREEN,   ""),
    logging.WARNING:  (_YELLOW,  _YELLOW),
    logging.ERROR:    (_RED,     _RED),
    logging.CRITICAL: (_RED,     _RED),
}


def _color_enabled(stream: object) -> bool:
    """Decide whether to emit ANSI color codes.

    Precedence (highest wins):

      1. ``NO_COLOR`` non-empty     -> off  (https://no-color.org/ spec
                                            mandates it wins over everything)
      2. ``MINIINFER_LOG_COLOR=0``  -> off
      3. ``MINIINFER_LOG_COLOR=1``  -> on
      4. ``CLICOLOR_FORCE=1``       -> on
      5. ``TERM=dumb``              -> off
      6. stream.isatty()            -> on, else off
    """
    # Per no-color.org: NO_COLOR always wins, even over explicit "force on".
    if os.environ.get("NO_COLOR", "") != "":
        return False
    forced = os.environ.get("MINIINFER_LOG_COLOR", "").strip().lower()
    if forced in ("0", "false", "no", "off"):
        return False
    if forced in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("CLICOLOR_FORCE", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("TERM", "") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class _MiniInferFormatter(logging.Formatter):
    """Emits vLLM-style lines: ``(Role pid=X) LEVEL MM-DD HH:MM:SS [file:line] msg``.

    When ``use_color`` is True, the line is split into colored spans:

      * ``(Role pid=N)``     — magenta, bold (process tag)
      * ``LEVEL``            — colored per severity (see ``_LEVEL_COLORS``)
      * ``MM-DD HH:MM:SS``   — dim
      * ``[file:line]``      — cyan
      * message              — default for INFO, level-colored otherwise
    """

    def __init__(self, use_color: bool = False):
        super().__init__()
        self._pid = os.getpid()
        self._use_color = use_color

    def _wrap(self, text: str, color: str) -> str:
        if not self._use_color or not color:
            return text
        return f"{color}{text}{_RESET}"

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%m-%d %H:%M:%S")
        # Truncate to basename — vLLM uses e.g. ``model_runner.py``, not the
        # full path. Fall back to ``<unknown>`` if pathname is missing (rare
        # but possible with logger.makeRecord).
        fname = os.path.basename(record.pathname) if record.pathname else "<unknown>"

        prefix_color, msg_color = _LEVEL_COLORS.get(
            record.levelno, (_GREEN, "")
        )

        proc_tag = self._wrap(f"({_ROLE} pid={self._pid})", _BOLD + _MAGENTA)
        level = self._wrap(f"{record.levelname:<5}", _BOLD + prefix_color)
        timestamp = self._wrap(ts, _DIM)
        location = self._wrap(f"[{fname}:{record.lineno}]", _CYAN)
        message = self._wrap(record.getMessage(), msg_color)

        return f"{proc_tag} {level} {timestamp} {location} {message}"


def set_process_role(role: str) -> None:
    """Override the per-process tag (e.g. ``Worker-1`` for TP rank > 0)."""
    global _ROLE
    _ROLE = role


def get_logger(name: str = "miniinfer") -> logging.Logger:
    """Return a configured logger.

    The first call installs a single ``StreamHandler`` on the named logger
    with INFO level; subsequent calls just attach the same handler set
    (so child modules share formatting).
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger
    logger.setLevel(logging.INFO)
    # Don't propagate to root — we own the formatting.
    logger.propagate = False
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_MiniInferFormatter(use_color=_color_enabled(sys.stdout)))
    logger.addHandler(handler)
    _CONFIGURED.add(name)
    return logger
