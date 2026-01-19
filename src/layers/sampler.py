import torch
from typing import Callable
from .linear import softmax
import torch.nn as nn
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SamplingBatchInfo:
    temperature: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    # Grammar-based sampling
    vocab_size: int
    grammars: Optional[List] = None
    vocab_mask: Optional[torch.Tensor] = None
    apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    has_custom_logits_processor: bool = False


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ) -> torch.Tensor:
        if sampling_batch_info.has_custom_logits_processor:
            apply_custom_logits_processor(logits, sampling_batch_info)

        return logits

    def forward(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """
        对 logits 进行采样，返回采样的 token ids

        Args:
            logits: [num_seqs, vocab_size]
            sampling_batch_info: 每个序列的采样信息

        Returns:
            采样的 token ids: [num_seqs, 1]
        """
        logits = self._preprocess_logits(logits, sampling_batch_info)

        batch_size = logits.shape[0]
        batch_next_token_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=logits.device
        )

        for i in range(batch_size):
            logprobs = torch.log_softmax(logits[i], dim=-1)
            temp = sampling_batch_info.temperature[i].item()
            top_p = sampling_batch_info.top_ps[i].item()
            top_k = sampling_batch_info.top_ks[i].item()

            sampler_fn = make_sampler(temp, top_p, top_k)
            next_token = sampler_fn(logprobs)  # [1]
            batch_next_token_ids[i, 0] = next_token

        return batch_next_token_ids  # [num_seqs, 1]


def apply_custom_logits_processor(
    logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
) -> torch.Tensor:
    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, 1)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )


def greedy_sample(logprobs: torch.Tensor):
    return torch.argmax(logprobs, axis=-1, keepdim=True)


def temperature_sample(logprobs: torch.Tensor, temp: float):
    probs = softmax(logprobs, axis=-1)
    idxs = torch.multinomial(probs, num_samples=1)
    return idxs


def top_k_sample(logprobs: torch.Tensor, k: int):
    topk_logprobs, _ = torch.topk(logprobs, k=k, dim=-1)
    min_topk_logprob = topk_logprobs[..., -1, None]
    logprobs = torch.where(
        logprobs < min_topk_logprob,
        torch.full_like(logprobs, float("-inf")),
        logprobs,
    )
    probs = softmax(logprobs, axis=-1)
    idxs = torch.multinomial(probs, num_samples=1)

    return idxs


def top_p_sample(logprobs: torch.Tensor, p: float):
    # 1. sort the logprobs from largest to smallest
    sorted_logprobs, sorted_indices = torch.sort(logprobs, descending=True, dim=-1)
    # 2. compute cumulative probabilities
    cumulative_probs = torch.cumsum(softmax(sorted_logprobs, axis=-1), dim=-1)
    # 3. truncate tokens with cumulative prob above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep the first token above the threshold
    if sorted_indices_to_remove[..., 1:].any():
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = (
            False  # like [False, False, True(keep last), True, True...]
        )

    indices_to_remove = torch.zeros_like(logprobs, dtype=torch.bool).scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    logprobs = torch.where(
        indices_to_remove,
        torch.full_like(logprobs, float("-inf")),
        logprobs,
    )
    probs = softmax(logprobs, axis=-1)
    idxs = torch.multinomial(probs, num_samples=1)

    return idxs


SAMPLE_IMPLEMENTATIONS: dict[str, Callable] = {
    "greedy": greedy_sample,
    "temperature": temperature_sample,
    "top_k": top_k_sample,
    "top_p": top_p_sample,
}


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        """This way will implement temperature sampling, top-k sampling, and top-p (nucleus) sampling."""
        if temp == 0:
            return greedy_sample(logprobs)
        else:
            logprobs = logprobs / temp
            if top_k is not None and top_k > 0:
                idxs = top_k_sample(logprobs, top_k)
            elif top_p < 1.0:
                idxs = top_p_sample(logprobs, top_p)
            else:
                idxs = temperature_sample(logprobs, temp)
            return idxs

    return sample
