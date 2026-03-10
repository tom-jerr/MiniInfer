"""Attention backend exports."""

from .base_backend import AttentionBackend

__all__ = [
  "AttentionBackend",
  "FlashAttention2Backend",
  "FlashAttention2Metadata",
  "FlashAttention3Backend",
  "FlashAttention3Metadata",
]


def __getattr__(name: str):
  if name in {
    "FlashAttention2Backend",
    "FlashAttention2Metadata",
    "FlashAttention3Backend",
    "FlashAttention3Metadata",
  }:
    from .flashattention_backend import (
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
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
