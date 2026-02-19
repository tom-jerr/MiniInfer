"""
GSM8K 集成测试：准确率 + 性能指标

覆盖指标：
1. Accuracy
2. Output Throughput (token/s)
3. TTFT (Time To First Token)
4. TPOT (Time Per Output Token, excluding first token)

运行方式：
    pytest tests/test_gsm8k_metrics.py -v -s

可选环境变量：
    MINIINFER_GSM8K_MODEL=Qwen/Qwen2-0.5B-Instruct
    MINIINFER_GSM8K_DATA_PATH=/path/to/gsm8k_test.jsonl
    MINIINFER_GSM8K_NUM_QUESTIONS=32
    MINIINFER_GSM8K_NUM_SHOTS=5
    MINIINFER_GSM8K_MAX_NEW_TOKENS=256

    MINIINFER_GSM8K_MIN_ACCURACY=0.35
    MINIINFER_GSM8K_MIN_THROUGHPUT=40
    MINIINFER_GSM8K_MAX_TTFT=2.0
    MINIINFER_GSM8K_MAX_TPOT=0.08
"""

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from miniinfer.engine.llm_engine import LLMEngine
from miniinfer.utils.sampling_params import SamplingParams

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
INVALID = -9999999.0
ANSWER_FORMAT_INSTRUCTION = (
    "Solve the math problem step by step. "
    'The final line must be exactly in this format: "#### <number>".'
)


def _model_exists_local(model_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_name, local_files_only=True)
        return True
    except Exception:
        return False


def _download_or_get_gsm8k(data_path: Optional[str]) -> str:
    if data_path:
        if not Path(data_path).exists():
            pytest.skip(f"GSM8K data not found: {data_path}")
        return data_path

    cache_dir = Path.home() / ".cache" / "miniinfer"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_file = cache_dir / "gsm8k_test.jsonl"
    if cached_file.exists():
        return str(cached_file)

    try:
        urllib.request.urlretrieve(GSM8K_URL, cached_file)
    except Exception as exc:
        pytest.skip(
            "Cannot download GSM8K test split in current environment. "
            f"Please set MINIINFER_GSM8K_DATA_PATH. Error: {exc}"
        )
    return str(cached_file)


def _read_jsonl(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_answer_value(text: str) -> float:
    try:
        if "####" in text:
            text = text.split("####", 1)[1]
        numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
        if not numbers:
            return INVALID
        return float(numbers[-1])
    except Exception:
        return INVALID


def _extract_pred_value(text: str) -> float:
    try:
        if "####" not in text:
            return INVALID
        after_marker = text.split("####", 1)[1]
        numbers = re.findall(r"-?\d+\.?\d*", after_marker.replace(",", ""))
        if not numbers:
            return INVALID
        return float(numbers[0])
    except Exception:
        return INVALID


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return None
    return float(value)


def _prepare_prompts_and_labels(
    rows: List[Dict[str, str]], num_shots: int, num_questions: int
) -> Tuple[List[str], List[float], List[str], List[str]]:
    if len(rows) < num_shots + num_questions:
        pytest.skip(
            f"Not enough GSM8K rows: need {num_shots + num_questions}, got {len(rows)}"
        )

    few_shot_parts = []
    for i in range(num_shots):
        few_shot_parts.append(
            f"Question: {rows[i]['question']}\nAnswer: {rows[i]['answer']}\n"
        )
    few_shot = "\n".join(few_shot_parts)

    prompts: List[str] = []
    labels: List[float] = []
    questions: List[str] = []
    gold_answers: List[str] = []
    for i in range(num_shots, num_shots + num_questions):
        prompt = (
            f"{ANSWER_FORMAT_INSTRUCTION}\n\n"
            f"{few_shot}\nQuestion: {rows[i]['question']}\nAnswer: "
        )
        label = _parse_answer_value(rows[i]["answer"])
        if label == INVALID:
            pytest.skip(f"Invalid gold answer format at row {i}")
        prompts.append(prompt)
        labels.append(label)
        questions.append(rows[i]["question"])
        gold_answers.append(rows[i]["answer"])

    return prompts, labels, questions, gold_answers


def _run_stream_eval(
    engine: LLMEngine,
    prompts: List[str],
    labels: List[float],
    questions: List[str],
    gold_answers: List[str],
    max_new_tokens: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, int]]:
    sampling_params = [
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        for _ in range(len(prompts))
    ]

    req_id_to_index: Dict[int, int] = {}
    req_start_at: Dict[int, float] = {}
    output_texts: Dict[int, str] = {}
    output_token_counts: Dict[int, int] = {}
    first_token_at: Dict[int, float] = {}
    last_token_at: Dict[int, float] = {}
    finish_reason_map: Dict[int, Optional[str]] = {}
    finished_req_ids = set()

    for idx, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
        req_id = engine.add_request(prompt, sp)
        req_id_to_index[req_id] = idx
        req_start_at[req_id] = time.perf_counter()
        output_texts[req_id] = ""
        output_token_counts[req_id] = 0

    start = time.perf_counter()
    while not engine.is_finished():
        step_output = engine.step()
        now = time.perf_counter()
        for output in step_output.outputs:
            req_id = output.request_id
            if req_id not in first_token_at:
                first_token_at[req_id] = now
            last_token_at[req_id] = now
            output_texts[req_id] = output.full_text
            output_token_counts[req_id] = len(output.output_token_ids)
            if output.finished:
                finish_reason_map[req_id] = output.finish_reason
            if output.finished:
                finished_req_ids.add(req_id)
    end = time.perf_counter()

    if len(finished_req_ids) != len(prompts):
        raise AssertionError(
            f"Expected {len(prompts)} finished requests, got {len(finished_req_ids)}"
        )

    finished_req_meta: Dict[int, Any] = {}
    for req in getattr(engine.scheduler, "finished_reqs", []):
        finished_req_meta[int(req.req_id)] = req

    ordered_req_ids = sorted(
        req_id_to_index.keys(), key=lambda rid: req_id_to_index[rid]
    )
    ordered_texts = [output_texts[rid] for rid in ordered_req_ids]
    ordered_token_counts = [output_token_counts[rid] for rid in ordered_req_ids]

    preds = [_extract_pred_value(text) for text in ordered_texts]
    correct = [
        abs(pred - label) < 1e-6 if pred != INVALID and label != INVALID else False
        for pred, label in zip(preds, labels)
    ]
    invalid = [pred == INVALID for pred in preds]

    latency = end - start
    total_output_tokens = sum(ordered_token_counts)
    output_throughput = total_output_tokens / latency if latency > 0 else 0.0

    ttft_values = [
        first_token_at[rid] - req_start_at[rid]
        for rid in ordered_req_ids
        if rid in first_token_at
    ]
    ttft_mean = _mean(ttft_values)

    tpot_values = []
    for rid, token_count in zip(ordered_req_ids, ordered_token_counts):
        if token_count <= 1:
            continue
        if rid not in first_token_at or rid not in last_token_at:
            continue
        tpot_values.append(
            (last_token_at[rid] - first_token_at[rid]) / float(token_count - 1)
        )
    tpot_mean = _mean(tpot_values)

    metrics = {
        "accuracy": _mean([1.0 if c else 0.0 for c in correct]),
        "invalid_rate": _mean([1.0 if x else 0.0 for x in invalid]),
        "latency_s": latency,
        "total_output_tokens": float(total_output_tokens),
        "output_throughput_tok_s": output_throughput,
        "ttft_mean_s": ttft_mean,
        "tpot_mean_s": tpot_mean,
    }

    case_reports: List[Dict[str, Any]] = []
    reason_counter: Dict[str, int] = {}
    for i, (rid, pred, label, text, token_count) in enumerate(
        zip(ordered_req_ids, preds, labels, ordered_texts, ordered_token_counts)
    ):
        is_invalid = pred == INVALID
        is_correct = (not is_invalid) and (abs(pred - label) < 1e-6)
        has_hash_marker = "####" in text
        finish_reason = finish_reason_map.get(rid)
        ever_retracted = bool(
            getattr(finished_req_meta.get(rid), "ever_retracted", False)
        )

        if is_invalid:
            if not has_hash_marker:
                if token_count >= max_new_tokens:
                    reason = "invalid:no_####_marker_and_hit_max_tokens"
                else:
                    reason = "invalid:no_####_marker"
            else:
                reason = "invalid:has_####_but_no_number"
        elif is_correct:
            reason = "correct"
        else:
            reason = "wrong_number:has_####_marker"

        reason_counter[reason] = reason_counter.get(reason, 0) + 1
        case_reports.append(
            {
                "index": i,
                "request_id": rid,
                "question": questions[i],
                "gold_answer_text": gold_answers[i],
                "gold_value": label,
                "pred_value": None if is_invalid else pred,
                "is_invalid": is_invalid,
                "is_correct": is_correct,
                "reason": reason,
                "has_hash_marker": has_hash_marker,
                "finish_reason": finish_reason,
                "ever_retracted": ever_retracted,
                "output_tokens": token_count,
                "model_output": text,
            }
        )

    return metrics, case_reports, reason_counter


def _print_case_reports(
    case_reports: List[Dict[str, Any]],
    reason_counter: Dict[str, int],
) -> None:
    print("\nPer-case analysis:")
    for case in case_reports:
        print("\n" + "-" * 80)
        print(f"Case #{case['index']}")
        print(f"Question: {case['question']}")
        print(f"Gold answer text: {case['gold_answer_text']}")
        print(f"Gold value: {case['gold_value']}")
        print(
            f"Pred value: "
            f"{'INVALID' if case['pred_value'] is None else case['pred_value']}"
        )
        print(
            f"Correct={case['is_correct']}, Invalid={case['is_invalid']}, "
            f"Reason={case['reason']}"
        )
        print(
            f"Finish reason={case['finish_reason']}, "
            f"Ever retracted={case['ever_retracted']}, "
            f"Output tokens={case['output_tokens']}, "
            f"Has #### marker={case['has_hash_marker']}"
        )
        print("Model output:")
        print(case["model_output"])

    print("\nReason summary:")
    for reason, count in sorted(reason_counter.items(), key=lambda x: x[0]):
        print(f"  {reason}: {count}")


def test_gsm8k_accuracy_throughput_ttft_tpot():
    model_name = os.getenv("MINIINFER_GSM8K_MODEL", "Qwen/Qwen2-0.5B-Instruct")
    num_questions = _env_int("MINIINFER_GSM8K_NUM_QUESTIONS", 5)
    num_shots = _env_int("MINIINFER_GSM8K_NUM_SHOTS", 5)
    max_new_tokens = _env_int("MINIINFER_GSM8K_MAX_NEW_TOKENS", 256)
    data_path = os.getenv("MINIINFER_GSM8K_DATA_PATH")

    if not torch.cuda.is_available():
        pytest.skip("GSM8K metrics test requires CUDA.")
    if not _model_exists_local(model_name):
        pytest.skip(
            f"Model {model_name} not found in local cache. "
            "Please pre-download or change MINIINFER_GSM8K_MODEL."
        )

    gsm8k_path = _download_or_get_gsm8k(data_path)
    rows = _read_jsonl(gsm8k_path)
    prompts, labels, questions, gold_answers = _prepare_prompts_and_labels(
        rows, num_shots, num_questions
    )

    llm = LLMEngine(
        model=model_name,
        enable_chunked_prefill=False,
        use_chat_template=False,
    )
    try:
        llm.generate(
            ["Warmup"],
            SamplingParams(temperature=0.0, max_tokens=2),
            use_tqdm=False,
        )
        metrics, case_reports, reason_counter = _run_stream_eval(
            llm,
            prompts,
            labels,
            questions,
            gold_answers,
            max_new_tokens=max_new_tokens,
        )
    finally:
        llm.stop()

    print("\nGSM8K metrics:")
    print(json.dumps(metrics, indent=2))
    _print_case_reports(case_reports, reason_counter)

    assert metrics["accuracy"] >= 0.0
    assert metrics["output_throughput_tok_s"] > 0.0
    assert metrics["ttft_mean_s"] >= 0.0
    assert metrics["tpot_mean_s"] >= 0.0

    min_acc = _env_float("MINIINFER_GSM8K_MIN_ACCURACY")
    min_throughput = _env_float("MINIINFER_GSM8K_MIN_THROUGHPUT")
    max_ttft = _env_float("MINIINFER_GSM8K_MAX_TTFT")
    max_tpot = _env_float("MINIINFER_GSM8K_MAX_TPOT")

    if min_acc is not None:
        assert metrics["accuracy"] >= min_acc
    if min_throughput is not None:
        assert metrics["output_throughput_tok_s"] >= min_throughput
    if max_ttft is not None:
        assert metrics["ttft_mean_s"] <= max_ttft
    if max_tpot is not None:
        assert metrics["tpot_mean_s"] <= max_tpot


if __name__ == "__main__":
    test_gsm8k_accuracy_throughput_ttft_tpot()
