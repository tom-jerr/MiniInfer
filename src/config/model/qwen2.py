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
        torch_dtype: str = "float32",
        **kwargs,
    ):
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.torch_dtype = torch_dtype

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


class Qwen2Config(PretrainedConfig):
    """
    Qwen2 模型配置类

    Args:
        vocab_size: 词汇表大小
        hidden_size: 隐藏层维度
        intermediate_size: MLP 中间层维度
        num_hidden_layers: Transformer 层数
        num_attention_heads: 注意力头数
        num_key_value_heads: KV 注意力头数(用于 GQA)
        hidden_act: 激活函数类型
        max_position_embeddings: 最大位置编码长度
        initializer_range: 权重初始化范围
        rms_norm_eps: RMSNorm 的 epsilon 值
        rope_theta: RoPE 的 theta 参数
        rope_scaling: RoPE 缩放配置
        attention_dropout: 注意力 dropout 概率
        tie_word_embeddings: 是否共享输入输出词嵌入
        use_cache: 是否使用 KV cache
        torch_dtype: 模型精度类型
    """

    model_type = "qwen2"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        use_cache=True,
        torch_dtype="float32",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_config(cls, hf_config):
        """从 HuggingFace 配置对象创建 Qwen2Config"""
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            hidden_act=getattr(hf_config, "hidden_act", "silu"),
            max_position_embeddings=hf_config.max_position_embeddings,
            initializer_range=getattr(hf_config, "initializer_range", 0.02),
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            rope_scaling=getattr(hf_config, "rope_scaling", None),
            attention_dropout=getattr(hf_config, "attention_dropout", 0.0),
            tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
            use_cache=getattr(hf_config, "use_cache", True),
            torch_dtype=str(hf_config.torch_dtype).split(".")[-1]
            if hasattr(hf_config, "torch_dtype")
            else "float32",
        )


# 预定义的模型配置
class TestQwen2Config:
    """Qwen2 系列模型的预定义配置"""

    @staticmethod
    def qwen2_0_5b() -> Qwen2Config:
        """Qwen2-0.5B 配置"""
        return Qwen2Config(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )

    @staticmethod
    def qwen2_1_5b() -> Qwen2Config:
        """Qwen2-1.5B 配置"""
        return Qwen2Config(
            vocab_size=151936,
            hidden_size=1536,
            intermediate_size=8960,
            num_hidden_layers=28,
            num_attention_heads=12,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=True,
        )

    @staticmethod
    def qwen2_7b() -> Qwen2Config:
        """Qwen2-7B 配置"""
        return Qwen2Config(
            vocab_size=152064,
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        )


__all__ = ["PretrainedConfig", "Qwen2Config", "TestQwen2Config"]
