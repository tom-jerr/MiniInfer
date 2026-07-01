"""权重加载：多线程 page-cache 预热 + 流式读取。

移植自 ``nano-vllm/nanovllm/utils/loader.py`` 的核心思想，但保留
MiniInfer 的「合并 state_dict」对外接口（dense 模型的 ``load_weights``
签名不变），只把读取阶段改成多线程：

1. **Prefetch（后台线程池）** —— 一组后台线程把每个 shard 顺序地按
   16MiB 大块读一遍。这一步并不保留字节，只是为了**预热 OS page
   cache**，使后续 ``safe_open`` 的随机 mmap 访问命中内存而非磁盘。
   顺序大块读远快于 ``safe_open`` 的惰性 mmap，在 NFS/Lustre 上尤其
   明显。
2. **逐 tensor 流式读取** —— 主线程对每个 shard ``safe_open`` 后用
   ``get_tensor(name)`` 一次取一个 tensor 路由进 state_dict，同时只
   实化单个 tensor，峰值内存低。Prefetch 在后台与该迭代重叠。

为什么不直接多线程整文件 ``load_file``？瓶颈是磁盘带宽，几个并发读
者就饱和了；再多的线程只会增加争用与（每个都持有整 shard 的）内存
压力。Prefetch + 逐 tensor 读攻克的正是真正瓶颈（冷 page cache +
随机 mmap 访问）。

MoE 兼容：``_merge_state_dict`` 会跳过 ``.experts.`` 路径下的键，使
per-expert 权重不参与 qkv/gate_up 合并，留给 MoE 模块自行堆叠。
"""
from __future__ import annotations

import concurrent.futures
import glob
import os
import threading
from collections.abc import Generator
from typing import Dict

import safetensors
import torch
from huggingface_hub import snapshot_download
from tqdm.asyncio import tqdm

# 延迟取 logger，避免 loader（被 layers/models 早期导入）与 miniinfer.utils
# 之间的静态循环导入（utils/__init__ 会拉起 model_utils -> models）。
def _log():
  from miniinfer.utils import get_logger
  return get_logger("miniinfer")


class DisabledTqdm(tqdm):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs, disable=True)


class WeightLoaderMixin:
  """权重加载基类，规范化权重填充逻辑"""

  def load_weights(self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype) -> None:
    raise NotImplementedError("Each module must implement its own weight loading logic.")

  def _to_device(self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype):
    return tensor.to(device=device, dtype=dtype)


def _share_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
  # TODO(lzy): support tp
  pass


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
  """把 HF 的 q/k/v、gate/up 合并成融合权重。

  注意：``.experts.`` 路径下的 per-expert 键（MoE）会**原样保留**——
  它们的 gate_proj/up_proj 由 MoE 模块自行堆叠进 w13_weight，不能
  在这里被合并成 gate_up_proj（否则会破坏 per-expert 结构）。
  """
  merged_state_dict: Dict[str, torch.Tensor] = {}
  for key in list(state_dict.keys()):
    # MoE per-expert 权重跳过合并，保持原样。
    if ".experts." in key:
      merged_state_dict[key] = state_dict[key]
      continue
    if key.count(".q_proj"):
      q_proj = state_dict[key]
      k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
      v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
      new_key = key.replace(".q_proj", ".qkv_proj")
      merged_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
      del state_dict[key]
      del state_dict[key.replace(".q_proj", ".k_proj")]
      del state_dict[key.replace(".q_proj", ".v_proj")]
    elif key.count(".gate_proj"):
      gate_proj = state_dict[key]
      up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
      new_key = key.replace(".gate_proj", ".gate_up_proj")
      merged_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
      del state_dict[key]
      del state_dict[key.replace(".gate_proj", ".up_proj")]
    elif key.count(".k_proj") or key.count(".v_proj") or key.count(".up_proj"):
      continue
    else:
      merged_state_dict[key] = state_dict[key]
  return merged_state_dict


# ---------------------------------------------------------------------------
# 多线程权重读取器
# ---------------------------------------------------------------------------
# vLLM/nano-vllm 默认：8 个 prefetch 线程，16MiB 块。
DEFAULT_PREFETCH_THREADS = 8
DEFAULT_PREFETCH_BLOCK_SIZE = 16 * 1024 * 1024


def _shard_files(path: str) -> list[str]:
  files = sorted(glob.glob(f"{path}/*.safetensors"))
  if not files:
    raise RuntimeError(f"Cannot find any *.safetensors weights under {path!r}")
  return files


def _prefetch_file(path: str, block_size: int) -> None:
  """顺序按块读 ``path`` 以预热 OS page cache。"""
  with open(path, "rb", buffering=0) as f:
    while f.read(block_size):
      pass


def _start_prefetch(files: list[str], num_threads: int, block_size: int) -> threading.Thread | None:
  """启动后台 page-cache 预热；返回驱动线程。"""
  if not files:
    return None

  def _run() -> None:
    max_workers = min(num_threads, len(files))
    with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
      # 消费迭代器以观测异常；预热失败非致命（退化为冷读）。
      for _ in ex.map(lambda p: _prefetch_file(p, block_size), files):
        pass

  thread = threading.Thread(target=_run, name="weight-prefetch", daemon=True)
  thread.start()
  return thread


def _iter_weights(
  files: list[str],
  *,
  enable_prefetch: bool,
  num_prefetch_threads: int,
  prefetch_block_size: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
  """跨 shard 逐个 yield ``(weight_name, tensor)``。Prefetch 在后台重叠运行。"""
  prefetch_thread = None
  if enable_prefetch:
    prefetch_thread = _start_prefetch(files, num_prefetch_threads, prefetch_block_size)
  try:
    for st_file in files:
      with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
        for name in f.keys():  # noqa: SIM118
          yield name, f.get_tensor(name)
  finally:
    if prefetch_thread is not None:
      prefetch_thread.join()


def load_hf_weight(
  model_path: str,
  device: torch.device,
  *,
  enable_prefetch: bool = True,
  num_prefetch_threads: int = DEFAULT_PREFETCH_THREADS,
  prefetch_block_size: int = DEFAULT_PREFETCH_BLOCK_SIZE,
) -> Dict[str, torch.Tensor]:
  """加载 HF safetensors 权重为合并后的 state_dict。

  多线程后台预热 page cache，主线程流式逐 tensor 读取，最后做 qkv/gate_up
  合并并搬到 ``device``。
  """
  import time

  if os.path.isdir(model_path):
    hf_folder = model_path
  else:
    try:
      hf_folder = snapshot_download(
        model_path,
        allow_patterns=["*.safetensors"],
        tqdm_class=DisabledTqdm,
      )
    except Exception:
      raise ValueError(
        f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
      )

  files = _shard_files(hf_folder)
  log = _log()
  log.info(f"Starting to load model {model_path}")
  log.info(f"Loading {len(files)} safetensors shards with {num_prefetch_threads} prefetch threads")
  t0 = time.time()

  state_dict: Dict[str, torch.Tensor] = {}
  for name, tensor in _iter_weights(
    files,
    enable_prefetch=enable_prefetch,
    num_prefetch_threads=num_prefetch_threads,
    prefetch_block_size=prefetch_block_size,
  ):
    state_dict[name] = tensor

  state_dict = _merge_state_dict(state_dict)
  state_dict = {k: v.to(device=device) for k, v in state_dict.items()}

  log.info(f"Model loaded in {time.time() - t0:.2f}s ({len(state_dict)} tensors)")
  return state_dict
