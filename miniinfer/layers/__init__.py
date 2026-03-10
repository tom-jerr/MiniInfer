__all__ = [
  "apply_activation",
  "get_activation",
  "get_attention",
  "causal_mask",
  "AttentionImpl",
  "AttentionBackend",
  "FlashAttention2Backend",
  "FlashAttention2Metadata",
  "FlashAttention3Backend",
  "FlashAttention3Metadata",
  "VocabEmbedding",
  "LMHead",
  "linear",
  "softmax",
  "RMSNorm",
  "RotaryEmbedding",
  # "make_sampler",
  "Sampler",
]


def __getattr__(name: str):
  if name in {"apply_activation", "get_activation"}:
    from .activation import apply_activation, get_activation

    return {
      "apply_activation": apply_activation,
      "get_activation": get_activation,
    }[name]
  if name in {"get_attention", "causal_mask", "AttentionImpl"}:
    from .attention import get_attention, causal_mask, AttentionImpl

    return {
      "get_attention": get_attention,
      "causal_mask": causal_mask,
      "AttentionImpl": AttentionImpl,
    }[name]
  if name == "AttentionBackend":
    from .attention_backend import AttentionBackend

    return AttentionBackend
  if name in {
    "FlashAttention2Backend",
    "FlashAttention2Metadata",
    "FlashAttention3Backend",
    "FlashAttention3Metadata",
  }:
    from .attention_backend import (
      FlashAttention2Backend,
      FlashAttention2Metadata,
      FlashAttention3Backend,
      FlashAttention3Metadata,
    )

    return {
      "FlashAttention2Backend": FlashAttention2Backend,
      "FlashAttention2Metadata": FlashAttention2Metadata,
      "FlashAttention3Backend": FlashAttention3Backend,
      "FlashAttention3Metadata": FlashAttention3Metadata,
    }[name]
  if name in {"VocabEmbedding", "LMHead"}:
    from .embedding import LMHead, VocabEmbedding

    return {
      "VocabEmbedding": VocabEmbedding,
      "LMHead": LMHead,
    }[name]
  if name == "RMSNorm":
    from .layernorm import RMSNorm

    return RMSNorm
  if name in {"linear", "softmax"}:
    from .linear import linear, softmax

    return {
      "linear": linear,
      "softmax": softmax,
    }[name]
  if name == "RotaryEmbedding":
    from .position_encoding import RotaryEmbedding

    return RotaryEmbedding
  if name == "Sampler":
    from .sample import Sampler

    return Sampler
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
