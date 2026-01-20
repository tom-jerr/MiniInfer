import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniinfer.engine.request import batch_generate
from miniinfer.models.qwen2 import Qwen2Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
parser.add_argument("--batch-size", type=int, default=5)
parser.add_argument("--prefill-step", type=int, default=128)
parser.add_argument("--max-seq-len", type=int, default=512)
parser.add_argument("--max-new-tokens", type=int, default=100)

args = parser.parse_args()

shanghai_wikipedia = """
Shanghai[a] is a direct-administered municipality and the most populous urban area in China. The city is located on the Chinese shoreline on the southern estuary of the Yangtze River, with the Huangpu River flowing through it. The population of the city proper is the second largest in the world after Chongqing, with around 24.87 million inhabitants in 2023, while the urban area is the most populous in China, with 29.87 million residents. As of 2022, the Greater Shanghai metropolitan area was estimated to produce a gross metropolitan product (nominal) of nearly 13 trillion RMB ($1.9 trillion).[13] Shanghai is one of the world's major centers for finance, business and economics, research, science and technology, manufacturing, transportation, tourism, and culture. The Port of Shanghai is the world's busiest container port.
""".strip()

shanghai_wikipedia += "Based on the previous information, "

prompts = [
    shanghai_wikipedia + "Where is Shanghai?",
    shanghai_wikipedia + "How much is the population of Shanghai?",
    shanghai_wikipedia + "What is the GDP of Shanghai?",
    shanghai_wikipedia + "What is the population of Shanghai?",
    shanghai_wikipedia + "What is the second largest city proper in China?",
    shanghai_wikipedia + "What is Shanghai known for?",
    shanghai_wikipedia + "What are the rivers in Shanghai?",
    shanghai_wikipedia + "Shanghai is the major center for what?",
    "What is the capital of France?",
    "Where is New York City?",
    "Where is Tokyo?",
    "What is the capital of China?",
    "Where is Pittsburgh?",
    "Where is Vancouver?",
    "Where is Toronto?",
    "Give me a short introduction to large language model.",
]

# shuffle prompts
random.shuffle(prompts)


def main():
    print(f"Using PyTorch version with model: {args.model}")

    # Load model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the transformers model
    torch_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Wrap with our custom model
    tiny_llm_model = Qwen2Model(torch_model)

    # Prepare prompts with chat template
    encoded_prompts = []
    for idx, prompt in enumerate(prompts):
        print(f"Prompt {idx}: {prompt[:100]}...")  # Show first 100 chars

        # Apply chat template if tokenizer supports it
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                print(f"Warning: Could not apply chat template: {e}")
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        encoded_prompts.append(formatted_prompt)

    print("\nStarting batch generation with:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Prefill step: {args.prefill_step}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  Number of prompts: {len(encoded_prompts)}")

    # Run batch generation
    result = batch_generate(
        tiny_llm_model,
        tokenizer,
        encoded_prompts,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        prefill_step=args.prefill_step,
    )

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)

    # Sort results by prompt index for consistent output
    result.sort(key=lambda x: x[0])

    for prompt_idx, text in result:
        print(f"\n--- Prompt {prompt_idx} ---")
        print(f"Q: {prompts[prompt_idx]}")
        print(f"A: {text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
