import argparse
import random
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.engine.request import batch_generate
from src.models.qwen2 import Qwen2Model

def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen2 model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--prefill-step", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--input-len", type=int, default=128)
    
    args = parser.parse_args()

    print(f"Using PyTorch version with model: {args.model}")
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

    # Prepare prompts
    print(f"Generating {args.num_prompts} prompts with approx {args.input_len} tokens each...")
    prompts = []
    vocab_size = tokenizer.vocab_size
    
    random.seed(0)
    for _ in range(args.num_prompts):
        random_tokens = [random.randint(0, vocab_size - 1) for _ in range(args.input_len)]
        prompt_text = tokenizer.decode(random_tokens, skip_special_tokens=True)
        prompts.append(prompt_text)

    print("\nStarting benchmark with:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Prefill step: {args.prefill_step}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  Number of prompts: {len(prompts)}")

    start_time = time.time()

    # Run batch generation
    result = batch_generate(
        tiny_llm_model,
        tokenizer,
        prompts,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        prefill_step=args.prefill_step,
    )
    
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS:")
    print("=" * 80)

    total_tokens = 0
    for _, text in result:
        tokens = tokenizer(text, add_special_tokens=False).input_ids
        total_tokens += len(tokens)

    throughput = total_tokens / total_time if total_time > 0 else 0
    
    print(f"Total time: {total_time:.2f} s")
    print(f"Total generated tokens: {total_tokens}")
    print(f"Throughput: {throughput:.2f} tokens/s")
    print("=" * 80)

if __name__ == "__main__":
    main()
