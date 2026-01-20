from __future__ import annotations
from typing import Dict
import torch
import safetensors
from tqdm.asyncio import tqdm
import os
from huggingface_hub import snapshot_download
import glob


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


class WeightLoaderMixin:
    """权重加载基类，规范化权重填充逻辑"""

    def load_weights(
        self, state_dict: Dict[str, torch.Tensor], prefix: str, device, dtype
    ) -> None:
        raise NotImplementedError(
            "Each module must implement its own weight loading logic."
        )

    def _to_device(
        self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype
    ):
        return tensor.to(device=device, dtype=dtype)


def _share_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # TODO(lzy): support tp
    pass


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
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


def load_hf_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
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
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return _merge_state_dict(state_dict)
