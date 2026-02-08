# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

import time
from random import randint, seed

from miniinfer.utils.sampling_params import SamplingParams
from miniinfer.engine.llm_engine import LLMEngine

import logging

logging.basicConfig(
    level=logging.DEBUG,  # 想要 INFO 就改 INFO
    format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
    force=True,  # Python 3.8+：确保生效（避免被别的库提前配置）
)


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    # align the hyperparameters
    llm = LLMEngine(
        model="Qwen/Qwen2-0.5B-Instruct",
        enable_chunked_prefill=False,  # 启用 chunked prefill
        chunked_prefill_size=4096,  # 每个 chunk 最多 4096 tokens
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    llm.generate(
        ["Benchmark: "], SamplingParams(temperature=0.1)
    )  # to warm up flashinfer
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
