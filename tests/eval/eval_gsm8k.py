import argparse
import json
import time
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from miniinfer.engine.request import batch_generate
from miniinfer.models.qwen2 import Qwen2Model

INVALID = -9999999


def download_and_cache_file(url: str) -> str:
  """Download and cache the GSM8K dataset"""
  import urllib.request

  cache_dir = Path.home() / ".cache" / "miniinfer"
  cache_dir.mkdir(parents=True, exist_ok=True)

  filename = cache_dir / "gsm8k_test.jsonl"

  if not filename.exists():
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filename)
    print(f"Saved to {filename}")
  else:
    print(f"Using cached file: {filename}")

  return str(filename)


def read_jsonl(filename: str):
  """Read JSONL file line by line"""
  with open(filename, "r", encoding="utf-8") as f:
    for line in f:
      yield json.loads(line.strip())


def get_answer_value(answer_str: str) -> float:
  """Extract numerical answer from the answer string

  GSM8K answers are in format: "#### 42"
  """
  try:
    if "####" in answer_str:
      # Extract number after ####
      answer = answer_str.split("####")[1].strip()
      # Remove commas and convert to float
      answer = answer.replace(",", "")
      return float(answer)
    else:
      # Try to extract last number in the string
      import re

      numbers = re.findall(r"-?\d+\.?\d*", answer_str)
      if numbers:
        return float(numbers[-1].replace(",", ""))
      return INVALID
  except (ValueError, IndexError):
    return INVALID


def get_few_shot_examples(lines: list, num_shots: int = 5) -> str:
  """Construct few-shot examples"""
  examples = []
  for i in range(num_shots):
    question = lines[i]["question"]
    answer = lines[i]["answer"]
    examples.append(f"Question: {question}\nAnswer: {answer}\n")

  return "\n".join(examples)


def get_one_example(lines: list, index: int, include_answer: bool = False) -> str:
  """Get one question, optionally with answer"""
  question = lines[index]["question"]
  if include_answer:
    answer = lines[index]["answer"]
    return f"Question: {question}\nAnswer: {answer}"
  else:
    return f"Question: {question}\nAnswer: "


def extract_generated_answer(text: str) -> float:
  """Extract numerical answer from generated text"""
  try:
    # Look for #### pattern first
    if "####" in text:
      answer = text.split("####")[1].strip()
      # Extract first number after ####
      import re

      numbers = re.findall(r"-?\d+\.?\d*", answer)
      if numbers:
        return float(numbers[0].replace(",", ""))

    # Otherwise find last number in the text
    import re

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
      return float(numbers[-1].replace(",", ""))

    return INVALID
  except (ValueError, IndexError):
    return INVALID


def dump_state_text(filename: str, results: list):
  """Save results to file"""
  with open(filename, "w", encoding="utf-8") as f:
    for idx, (prompt_idx, text) in enumerate(results):
      f.write(f"=== Example {idx} (Prompt {prompt_idx}) ===\n")
      f.write(text)
      f.write("\n\n")
  print(f"Results saved to {filename}")


def run_eval(args):
  """Run GSM8K evaluation"""

  # Load data
  if args.data_path is None:
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    filename = download_and_cache_file(url)
  else:
    filename = args.data_path

  print(f"Loading data from {filename}...")
  lines = list(read_jsonl(filename))
  print(f"Loaded {len(lines)} examples")

  # Prepare questions and labels
  num_questions = min(args.num_questions, len(lines))
  num_shots = args.num_shots

  print(f"\nConstructing {num_shots}-shot prompts for {num_questions} questions...")
  few_shot_examples = get_few_shot_examples(lines, num_shots)

  questions = []
  labels = []
  prompts = []

  # Use different examples for few-shot and testing
  test_start_idx = num_shots
  for i in range(test_start_idx, test_start_idx + num_questions):
    question = get_one_example(lines, i, False)
    label = get_answer_value(lines[i]["answer"])

    # Construct full prompt with few-shot examples
    full_prompt = few_shot_examples + "\n" + question

    questions.append(question)
    labels.append(label)
    prompts.append(full_prompt)

  assert all(l != INVALID for l in labels), "Some labels are invalid!"

  print(f"\nExample prompt (first 500 chars):\n{prompts[0][:500]}...\n")

  # Load model
  print(f"Loading model {args.model}...")
  device = torch.device(args.device if torch.cuda.is_available() else "cpu")

  torch_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map=device,
  )
  tokenizer = AutoTokenizer.from_pretrained(args.model)

  # Wrap with our custom model
  tiny_llm_model = Qwen2Model(torch_model)
  tiny_llm_model.eval()

  tic = time.perf_counter()
  results = batch_generate(
    tiny_llm_model,
    tokenizer,
    prompts,
    max_seq_len=args.max_seq_len,
    batch_size=args.batch_size,
    prefill_step=args.prefill_step,
  )
  latency = time.perf_counter() - tic

  # Sort results by prompt index
  results.sort(key=lambda x: x[0])

  # Extract predictions
  preds = []
  for prompt_idx, generated_text in results:
    pred = extract_generated_answer(generated_text)
    preds.append(pred)

  # Compute metrics
  preds_array = np.array(preds)
  labels_array = np.array(labels)

  correct = preds_array == labels_array
  acc = np.mean(correct)
  invalid = np.mean(preds_array == INVALID)

  # Estimate token count (rough approximation)
  total_output_chars = sum(len(text) for _, text in results)
  estimated_tokens = total_output_chars / 4  # Rough estimate: 4 chars per token
  output_throughput = estimated_tokens / latency

  # Print results
  print("\n" + "=" * 60)
  print("EVALUATION RESULTS")
  print("=" * 60)
  print(f"Accuracy: {acc:.3f} ({np.sum(correct)}/{len(labels)} correct)")
  print(f"Invalid: {invalid:.3f} ({np.sum(preds_array == INVALID)}/{len(labels)} invalid)")
  print(f"Latency: {latency:.3f} s")
  print(f"Output throughput (est.): {output_throughput:.1f} token/s")
  print("=" * 60)

  # Dump detailed results
  output_file = args.output_file or "eval_gsm8k_output.txt"
  dump_state_text(output_file, results)

  return {
    "accuracy": acc,
    "invalid": invalid,
    "latency": latency,
    "output_throughput": output_throughput,
  }


def main():
  parser = argparse.ArgumentParser(description="Evaluate model on GSM8K dataset")

  # Model arguments
  parser.add_argument(
    "--model",
    type=str,
    default="Qwen/Qwen2-0.5B-Instruct",
    help="Model name or path",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
    help="Device to use",
  )

  # Data arguments
  parser.add_argument(
    "--data-path",
    type=str,
    default=None,
    help="Path to GSM8K data file (default: download from GitHub)",
  )
  parser.add_argument(
    "--num-questions", type=int, default=100, help="Number of questions to evaluate"
  )
  parser.add_argument("--num-shots", type=int, default=5, help="Number of few-shot examples")

  # Generation arguments
  parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
  parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length")
  parser.add_argument("--prefill-step", type=int, default=256, help="Prefill step size")
  parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=1024,
    help="Maximum new tokens to generate",
  )

  # Output arguments
  parser.add_argument(
    "--output-dir",
    type=str,
    default="output",
    help="Output directionary for output files",
  )
  parser.add_argument(
    "--output-file", type=str, default=None, help="Output file for detailed results"
  )

  args = parser.parse_args()

  # Run evaluation
  results = run_eval(args)

  # Save metrics to JSON
  metrics_file = args.output_dir + "eval_gsm8k_metrics.json"
  with open(metrics_file, "w") as f:
    json.dump(results, f, indent=2)
  print(f"\nMetrics saved to {metrics_file}")


def cli():
  """Command-line interface entry point"""
  main()


if __name__ == "__main__":
  main()
