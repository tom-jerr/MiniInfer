import sys
import os
import argparse
import traceback

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
  parser.add_argument(
    "--backend", type=str, default="flashinfer", choices=["flashinfer", "flash_attn"]
  )
  parser.add_argument("--page-size", type=int, default=16)
  parser.add_argument(
    "--prompt", type=str, default="Tell me a short story about a coding assistant."
  )
  parser.add_argument("--max-tokens", type=int, default=50)
  args = parser.parse_args()

  print(
    f"Initializing Engine with model={args.model}, backend={args.backend}, page_size={args.page_size}"
  )

  try:
    engine = LLMEngine(
      model=args.model,
      attention_backend=args.backend,
      page_size=args.page_size,
      max_num_seqs=16,
      trust_remote_code=True,
    )
  except Exception as e:
    print(f"Failed to initialize engine: {e}")
    traceback.print_exc()
    return

  try:
    print("Starting generation...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)

    # LLMEngine.generate returns a list of results
    outputs = engine.generate([args.prompt], sampling_params)

    for i, output in enumerate(outputs):
      print(f"\n--- Output {i} ---")
      print(f"Generated Text: {output['text']}")
      print(f"Number of tokens: {len(output['token_ids'])}")

  except Exception as e:
    print(f"Error during generation: {e}")
    traceback.print_exc()
  finally:
    if "engine" in locals():
      engine.stop()


if __name__ == "__main__":
  main()
