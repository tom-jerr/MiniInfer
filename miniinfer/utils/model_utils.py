from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from miniinfer.engine.request import Request


def shortcut_name_to_full_name(shortcut_name: str):
  lower_shortcut_name = shortcut_name.lower()
  if lower_shortcut_name == "qwen2-7b":
    return "Qwen/Qwen2-7B-Instruct"
  elif lower_shortcut_name == "qwen2-0.5b":
    return "Qwen/Qwen2-0.5B-Instruct"
  elif lower_shortcut_name == "qwen2-1.5b":
    return "Qwen/Qwen2-1.5B-Instruct"
  elif lower_shortcut_name == "qwen3-8b":
    return "mlx-community/Qwen3-8B-4bit"
  elif lower_shortcut_name == "qwen3-0.6b":
    return "mlx-community/Qwen3-0.6B-4bit"
  elif lower_shortcut_name == "qwen3-1.7b":
    return "mlx-community/Qwen3-1.7B-4bit"
  elif lower_shortcut_name == "qwen3-4b":
    return "mlx-community/Qwen3-4B-4bit"
  else:
    return shortcut_name


def make_model(model_name: str, torch_model, **kwargs):
  # Import here to avoid circular import
  from ..models import Qwen2Model

  model_name = shortcut_name_to_full_name(model_name)
  return Qwen2Model(torch_model, **kwargs)


def _print_progress(
  requests: list[Request | None],
  is_idle: list[bool],
  pending_prefill_request: Request | None,
  queue_size: int,
  progress_cnt: int,
  start_time: datetime,
):
  print(f"  --- {datetime.now() - start_time}")
  animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
  animation_frame = animation_frames[progress_cnt % len(animation_frames)]
  for i in range(len(requests)):
    if is_idle[i]:
      print(f"  Decode #{i}: idle", flush=True)
    else:
      text_preview = requests[i].text()[-80:].replace("\n", " ")
      print(
        f"{animation_frame} Decode [req {requests[i].prompt_idx}, {requests[i].offset}]: {text_preview}",
        flush=True,
      )
  if pending_prefill_request is not None:
    if pending_prefill_request.is_prefill_done:
      print(
        f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
        flush=True,
      )
      return
    precentage = (
      pending_prefill_request.offset / pending_prefill_request.prefill_tokens.size(-1)
    ) * 100
    print(
      f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {precentage:.2f}% ({pending_prefill_request.prefill_tokens.size(-1) - pending_prefill_request.offset} remaining tokens)",
      flush=True,
    )
  else:
    print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)
