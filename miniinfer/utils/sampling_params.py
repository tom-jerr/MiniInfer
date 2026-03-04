from dataclasses import dataclass


@dataclass
class SamplingParams:
  temperature: float = 0
  max_tokens: int = 1
  ignore_eos: bool = False
  # Sampling controls. Set `temperature=0` for greedy decoding.
  top_k: int = 0  # 0 disables top-k
  top_p: float = 1.0  # 1.0 disables top-p

  def __post_init__(self):
    if self.temperature < 0:
      raise ValueError("temperature must be >= 0")
    if self.max_tokens <= 0:
      raise ValueError("max_tokens must be > 0")
    if self.top_k is None:
      self.top_k = 0
    if self.top_k < 0:
      raise ValueError("top_k must be >= 0")
    if not (0.0 < float(self.top_p) <= 1.0):
      raise ValueError("top_p must be in (0, 1]")
