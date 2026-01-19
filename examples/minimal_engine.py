#!/usr/bin/env python3
"""
最小可运行的单进程 LLMEngine 示例

使用方式:
    python examples/minimal_engine.py --model /path/to/your/model

注意: model 必须是本地路径，不支持 HuggingFace Hub 直接下载
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engine.llm_engine import LLMEngine
from src.utils.sampling_params import SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Minimal LLMEngine Example")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Local path to the model directory",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=4,
        help="Maximum number of sequences in a batch",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Maximum model context length",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Max sequences: {args.max_num_seqs}")
    print(f"Max model length: {args.max_model_len}")
    print("-" * 50)

    # 测试 prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]

    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # 使用 context manager 运行
    # LLMEngine 接受 model 作为第一个参数，其他配置通过 kwargs 传入
    with LLMEngine(
        model=args.model,
        use_multiprocess=False,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
    ) as engine:
        print(f"\nGenerating responses for {len(prompts)} prompts...\n")

        outputs = engine.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        print("\n" + "=" * 50)
        print("Results:")
        print("=" * 50)

        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n[Prompt {i+1}]: {prompt}")
            print(f"[Response]: {output['text']}")
            print(f"[Token count]: {len(output['token_ids'])}")
            print("-" * 50)


def test_step_by_step(model_path: str):
    """
    演示如何使用 step-by-step API 进行更精细的控制
    """
    print("\n" + "=" * 50)
    print("Step-by-step API Demo")
    print("=" * 50)

    engine = LLMEngine(
        model=model_path,
        use_multiprocess=False,
        max_num_seqs=2,
        max_model_len=256,
    )
    engine.start()

    try:
        # 添加请求
        sampling_params = SamplingParams(max_tokens=32, temperature=0.7)

        req_id1 = engine.add_request("What is 2+2?", sampling_params)
        req_id2 = engine.add_request("Say hello in French.", sampling_params)

        print(f"Added requests: {req_id1[:8]}..., {req_id2[:8]}...")

        # 手动执行 step
        step_count = 0
        all_outputs = {}

        while not engine.is_finished():
            output, num_tokens = engine.step()
            step_count += 1

            if num_tokens > 0:
                print(f"Step {step_count}: Prefill {num_tokens} tokens")
            elif num_tokens < 0:
                print(f"Step {step_count}: Decode {-num_tokens} tokens")

            for seq_id, token_ids in output:
                all_outputs[seq_id] = engine.tokenizer.decode(token_ids)
                print(f"  -> Sequence {seq_id} finished")

        print(f"\nTotal steps: {step_count}")
        print("\nOutputs:")
        for seq_id, text in all_outputs.items():
            print(f"  {seq_id}: {text}")

    finally:
        engine.stop()


if __name__ == "__main__":
    main()

    # 如果想测试 step-by-step API，取消下面的注释并提供模型路径
    # test_step_by_step("/path/to/your/model")
