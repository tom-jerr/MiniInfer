"""
idea and most code from tranformers
"""

import json
from pathlib import Path
from typing import Any


class PretrainedConfig:
  """
  所有模型配置的基类，类似于 transformers.PretrainedConfig
  提供配置的通用接口和方法
  """

  model_type: str = ""

  def __init__(
    self,
    tie_word_embeddings: bool = False,
    use_cache: bool = True,
    dtype: str = "float32",
    **kwargs,
  ):
    self.tie_word_embeddings = tie_word_embeddings
    self.use_cache = use_cache
    self.dtype = dtype

    # 存储额外的参数
    for key, value in kwargs.items():
      setattr(self, key, value)

  def to_dict(self) -> dict[str, Any]:
    """将配置转换为字典"""
    output = {}
    for key, value in self.__dict__.items():
      if not key.startswith("_"):
        output[key] = value
    output["model_type"] = self.model_type
    return output

  @classmethod
  def from_dict(cls, config_dict: dict[str, Any], **kwargs):
    """从字典创建配置实例"""
    config = cls(**config_dict)
    for key, value in kwargs.items():
      setattr(config, key, value)
    return config

  def to_json_file(self, json_file_path: str):
    """将配置保存为 JSON 文件"""
    with open(json_file_path, "w", encoding="utf-8") as f:
      json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

  @classmethod
  def from_json_file(cls, json_file_path: str):
    """从 JSON 文件加载配置"""
    with open(json_file_path, encoding="utf-8") as f:
      config_dict = json.load(f)
    return cls.from_dict(config_dict)

  @classmethod
  def from_pretrained(cls, model_name_or_path: str):
    """
    从预训练模型路径或 HuggingFace 模型加载配置

    Args:
        model_name_or_path: 本地路径或 HuggingFace 模型名称

    Returns:
        配置实例
    """
    # 尝试从本地路径加载
    config_path = Path(model_name_or_path)
    if config_path.exists() and config_path.is_dir():
      config_file = config_path / "config.json"
      if config_file.exists():
        return cls.from_json_file(str(config_file))

    # 尝试从 HuggingFace 加载
    try:
      from transformers import AutoConfig

      hf_config = AutoConfig.from_pretrained(model_name_or_path)
      return cls.from_hf_config(hf_config)
    except ImportError:
      raise ImportError("需要安装 transformers 库才能从 HuggingFace 加载配置")
    except Exception as e:
      raise ValueError(f"无法从 {model_name_or_path} 加载配置: {e}")

  @classmethod
  def from_hf_config(cls, hf_config):
    """从 HuggingFace 配置对象创建配置"""
    raise NotImplementedError("子类需要实现 from_hf_config 方法")

  def __repr__(self):
    return f"{self.__class__.__name__} {self.to_dict()}"


__all__ = ["PretrainedConfig"]
