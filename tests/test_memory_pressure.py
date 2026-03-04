"""
显存压力测试

测试 prefill 和 decode 阶段在请求总 token 数超过 KV cache 容量时：
1. 不会 OOM（调度器正确收缩/分 chunk）
2. 所有请求最终能正常完成生成

场景设计:
  - test_prefill_exceeds_kv_cache:
      构造一批 prefill 请求，其 input token 总量远超 KV cache 容量，
      验证调度器通过分批 prefill + token 预算控制避免 OOM。

  - test_decode_exceeds_kv_cache:
      构造大量并发 decode 请求，使 running batch 的 KV cache 占用
      超过容量，验证 retract 机制正常工作且最终全部完成。

  - test_mixed_pressure:
      同时存在长 prefill + 大量短 decode，综合压力测试。

  - test_chunked_prefill_exceeds_memory:
      启用 chunked prefill 后，单条超长输入被正确拆 chunk 处理。

  - test_real_world_questions_smoke:
      使用真实自然语言问题进行并发生成，验证实际任务场景可稳定完成。

用法:
    pytest tests/test_memory_pressure.py -v -s
"""

import gc
import time
import logging
from random import randint, seed, shuffle

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


from miniinfer.utils.sampling_params import SamplingParams
from miniinfer.engine.llm_engine import LLMEngine

logging.basicConfig(
  level=logging.WARNING,
  format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
  force=True,
  # filename="memory_pressure_test.log",
  # filemode="w",
)
logger = logging.getLogger(__name__)

# ============ 全局配置 ============
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
REAL_WORLD_QUESTIONS = [
  "如果我要在 3 个月内从零学会 Python，请给一个每周学习计划。",
  "请解释什么是向量数据库，并给出一个实际应用场景。",
  "我有 100 万预算，想开一家 30 平米的咖啡店，先做哪些准备？",
  # "把这句话改写得更礼貌：你这个方案不行，重新做。",
  # "为什么飞机在高空飞行更省油？请用通俗语言解释。",
  # "我总是拖延，给我 5 条可执行的反拖延建议。",
  # "比较 REST 和 GraphQL 的主要区别，并说明各自适用场景。",
  # "用户反馈 App 很卡，作为工程师该如何系统排查性能问题？",
]


def _get_kv_cache_capacity(engine: LLMEngine) -> int:
  """获取引擎的 KV cache 总容量（token 数）"""
  return engine.kv_cache_mgr.size


def _warmup(engine: LLMEngine):
  """预热引擎，确保 FlashInfer / Triton kernel 已编译"""
  engine.generate(["Warmup"], SamplingParams(temperature=0.1, max_tokens=2))


def _decode_token(tokenizer, token_id: int) -> str:
  try:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)
  except Exception:
    return ""


def _token_piece(tokenizer, token_id: int) -> str:
  """返回 tokenizer 词表中的原始 token 片段（非 decode 后文本）。"""
  try:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
      return ""
    return str(token)
  except Exception:
    return ""


def _collect_hf_greedy_tokens(
  hf_model,
  prompt_token_ids: list[int],
  max_new_tokens: int,
  device: str,
) -> list[int]:
  """使用 transformers 做逐步 greedy 解码，返回每步选中的 token id。"""
  generated = list(prompt_token_ids)
  out_token_ids: list[int] = []
  with torch.no_grad():
    for _ in range(max_new_tokens):
      input_ids = torch.tensor([generated], dtype=torch.long, device=device)
      logits = hf_model(input_ids=input_ids).logits[0, -1]
      next_token = int(torch.argmax(logits, dim=-1).item())
      out_token_ids.append(next_token)
      generated.append(next_token)
  return out_token_ids


# ===================================================================
# Test 1: Prefill 总量超过 KV cache 容量
# ===================================================================
class TestPrefillExceedsMemory:
  """
  构造大量 prefill 请求，使所有 prompt 的 token 总量
  远超 KV cache 容量。调度器应通过以下机制防止 OOM：
    - PrefillAdder 中的 token 预算控制
    - 分批处理 waiting_queue（一次只取预算范围内的请求）
    - new_token_ratio 为 decode 预留空间
  """

  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=False,
    )
    _warmup(llm)
    yield llm

  def test_many_long_prefills(self, engine):
    """
    大量长 prompt 的 prefill 测试。
    总 input tokens >> KV cache 容量。
    所有请求应最终完成，不 OOM。
    """
    seed(42)
    kv_capacity = _get_kv_cache_capacity(engine)
    logger.info(f"KV cache capacity: {kv_capacity} tokens")

    # 构造请求：每条 prompt 512~1024 tokens，数量足以超出 KV cache 2 倍
    prompt_len_min, prompt_len_max = 512, 1024
    avg_prompt_len = (prompt_len_min + prompt_len_max) // 2
    # 目标：总 input tokens ≈ 2x KV cache 容量
    num_seqs = max(8, (kv_capacity * 2) // avg_prompt_len)
    # 限制最大数量，避免测试时间过长
    num_seqs = min(num_seqs, 128)

    output_tokens = 16  # 较短输出，只验证 prefill 压力

    prompt_token_ids = [
      [randint(0, 10000) for _ in range(randint(prompt_len_min, prompt_len_max))]
      for _ in range(num_seqs)
    ]
    sampling_params = [
      SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens)
      for _ in range(num_seqs)
    ]

    total_input_tokens = sum(len(p) for p in prompt_token_ids)
    logger.info(
      f"Prefill pressure test: {num_seqs} seqs, "
      f"total_input_tokens={total_input_tokens}, "
      f"kv_capacity={kv_capacity}, "
      f"ratio={total_input_tokens / kv_capacity:.2f}x"
    )

    t = time.time()
    results = engine.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    # 验证
    assert len(results) == num_seqs, f"Expected {num_seqs} results, got {len(results)}"
    for i, result in enumerate(results):
      assert len(result["token_ids"]) == output_tokens, (
        f"Request {i}: expected {output_tokens} output tokens, " f"got {len(result['token_ids'])}"
      )
      assert len(result["text"]) > 0, f"Request {i}: empty output text"

    total_output = sum(len(r["token_ids"]) for r in results)
    logger.info(
      f"Prefill pressure test PASSED: "
      f"{total_output} output tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 2: Decode 阶段并发请求超过 KV cache 容量
# ===================================================================
class TestDecodeExceedsMemory:
  """
  构造大量请求使 decode 阶段的 KV cache 占用超过容量。
  调度器应通过 retract 机制收缩 batch：
    - check_decode_mem 检测内存不足
    - 从 batch 尾部移除请求，放回 waiting_queue
    - 动态调高 new_token_ratio
  """

  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=False,
    )
    _warmup(llm)
    yield llm

  def test_many_concurrent_decodes(self, engine):
    """
    大量短 prompt + 中等长度输出。
    当所有请求同时进入 decode 阶段，
    占用的 KV cache 将超过容量。
    retract 机制应保证不 OOM 且全部完成。
    """
    seed(123)
    kv_capacity = _get_kv_cache_capacity(engine)
    logger.info(f"KV cache capacity: {kv_capacity} tokens")

    # 短 prompt (128~256 tokens)，较长输出 (128~512 tokens)
    # 使得 decode 阶段累计序列长度超过 KV cache
    prompt_len = 128
    output_len_min, output_len_max = 128, 512
    avg_total = prompt_len + (output_len_min + output_len_max) // 2

    # 目标：所有请求的 (prompt + output) 总长度 ≈ 2x KV cache
    num_seqs = max(8, (kv_capacity * 2) // avg_total)
    num_seqs = min(num_seqs, 256)

    prompt_token_ids = [[randint(0, 10000) for _ in range(prompt_len)] for _ in range(num_seqs)]
    sampling_params = [
      SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=randint(output_len_min, output_len_max),
      )
      for _ in range(num_seqs)
    ]

    total_input = sum(len(p) for p in prompt_token_ids)
    total_max_output = sum(sp.max_tokens for sp in sampling_params)
    total_peak = total_input + total_max_output
    logger.info(
      f"Decode pressure test: {num_seqs} seqs, "
      f"total_input={total_input}, total_max_output={total_max_output}, "
      f"peak_kv={total_peak}, kv_capacity={kv_capacity}, "
      f"ratio={total_peak / kv_capacity:.2f}x"
    )

    t = time.time()
    results = engine.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    # 验证
    assert len(results) == num_seqs, f"Expected {num_seqs} results, got {len(results)}"
    for i, result in enumerate(results):
      expected_len = sampling_params[i].max_tokens
      assert len(result["token_ids"]) == expected_len, (
        f"Request {i}: expected {expected_len} output tokens, " f"got {len(result['token_ids'])}"
      )

    total_output = sum(len(r["token_ids"]) for r in results)
    logger.info(
      f"Decode pressure test PASSED: "
      f"{total_output} output tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 3: 混合压力 — 长 prefill + 大量 decode 同时进行
# ===================================================================
class TestMixedPressure:
  """
  同时存在:
    - 长 prompt 请求（prefill 压力）
    - 大量短 prompt + 长输出请求（decode 压力）
  验证调度器在混合场景下的正确性。
  """

  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=False,
    )
    _warmup(llm)
    yield llm

  def test_mixed_long_prefill_and_decode(self, engine):
    """
    混合长/短 prompt + 不同输出长度，
    总 KV 需求 >> 容量。
    """
    seed(456)
    kv_capacity = _get_kv_cache_capacity(engine)
    logger.info(f"KV cache capacity: {kv_capacity} tokens")

    prompts = []
    params = []

    # 10% 长 prompt (512~1024 tokens) + 短输出
    num_long = max(2, kv_capacity // 2048)
    num_long = min(num_long, 16)
    for _ in range(num_long):
      length = randint(512, 1024)
      prompts.append([randint(0, 10000) for _ in range(length)])
      params.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16))

    # 90% 短 prompt (64~128 tokens) + 中等输出 (64~256 tokens)
    num_short = max(8, kv_capacity // 512)
    num_short = min(num_short, 128)
    for _ in range(num_short):
      length = randint(64, 128)
      prompts.append([randint(0, 10000) for _ in range(length)])
      params.append(
        SamplingParams(
          temperature=0.6,
          ignore_eos=True,
          max_tokens=randint(64, 256),
        )
      )

    num_seqs = len(prompts)
    total_input = sum(len(p) for p in prompts)
    total_max_output = sum(sp.max_tokens for sp in params)
    logger.info(
      f"Mixed pressure test: {num_seqs} seqs "
      f"({num_long} long + {num_short} short), "
      f"total_input={total_input}, total_max_output={total_max_output}, "
      f"kv_capacity={kv_capacity}"
    )

    t = time.time()
    results = engine.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - t

    assert len(results) == num_seqs
    for i, result in enumerate(results):
      expected_len = params[i].max_tokens
      assert len(result["token_ids"]) == expected_len, (
        f"Request {i}: expected {expected_len} tokens, " f"got {len(result['token_ids'])}"
      )

    total_output = sum(len(r["token_ids"]) for r in results)
    logger.info(
      f"Mixed pressure test PASSED: "
      f"{total_output} tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 4: Chunked Prefill — 单条超长输入
# ===================================================================
class TestChunkedPrefillExceedsMemory:
  """
  启用 chunked prefill 后，单条超长输入应被
  自动拆分为多个 chunk 处理，而不 OOM。
  """

  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=True,
      chunked_prefill_size=2048,
    )
    _warmup(llm)
    yield llm

  def test_single_very_long_prefill(self, engine):
    """
    单条 prompt 长度接近 KV cache 容量的 50%，
    chunked prefill 应将其拆分为多个 chunk。
    """
    seed(789)
    kv_capacity = _get_kv_cache_capacity(engine)
    logger.info(f"KV cache capacity: {kv_capacity} tokens")

    # 单条 prompt 占 KV cache 的 50%
    prompt_len = min(kv_capacity // 2, 3072)  # 不超过 max_context_len
    output_tokens = 32

    prompt_token_ids = [[randint(0, 10000) for _ in range(prompt_len)]]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens)]

    logger.info(f"Chunked prefill test: prompt_len={prompt_len}, " f"kv_capacity={kv_capacity}")

    t = time.time()
    results = engine.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    assert len(results) == 1
    assert len(results[0]["token_ids"]) == output_tokens
    assert len(results[0]["text"]) > 0

    logger.info(f"Chunked prefill test PASSED: " f"{output_tokens} tokens in {elapsed:.2f}s")

  def test_multiple_long_prefills_chunked(self, engine):
    """
    多条长 prompt + chunked prefill，
    总 token 超过 KV cache 容量。
    """
    seed(101)
    kv_capacity = _get_kv_cache_capacity(engine)

    # 多条 prompt，每条 512~1024
    num_seqs = max(4, kv_capacity // 1024)
    num_seqs = min(num_seqs, 32)
    output_tokens = 16

    prompt_token_ids = [
      [randint(0, 10000) for _ in range(randint(512, 1024))] for _ in range(num_seqs)
    ]
    sampling_params = [
      SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens)
      for _ in range(num_seqs)
    ]

    total_input = sum(len(p) for p in prompt_token_ids)
    logger.info(
      f"Multiple chunked prefills: {num_seqs} seqs, "
      f"total_input={total_input}, kv_capacity={kv_capacity}"
    )

    t = time.time()
    results = engine.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    assert len(results) == num_seqs
    for i, result in enumerate(results):
      assert len(result["token_ids"]) == output_tokens, (
        f"Request {i}: expected {output_tokens} tokens, " f"got {len(result['token_ids'])}"
      )

    total_output = sum(len(r["token_ids"]) for r in results)
    logger.info(
      f"Multiple chunked prefills PASSED: "
      f"{total_output} tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 5: Chunked + Decode 混合压力
# ===================================================================
class TestChunkedMixedPrefillDecodePressure:
  """
  在 chunked prefill 开启时，构造同时包含：
    - 长 prompt（触发 chunked prefill）
    - 大量短 prompt + 长输出（decode 压力）
  验证 mixed 调度路径稳定性。
  """

  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=True,
      chunked_prefill_size=1024,
    )
    _warmup(llm)
    yield llm

  def test_chunked_mixed_prefill_decode_pressure(self, engine):
    seed(2026)
    kv_capacity = _get_kv_cache_capacity(engine)
    logger.info(f"KV cache capacity: {kv_capacity} tokens")

    prompts = []
    params = []

    # 长 prompt：触发 chunked prefill
    num_long = max(6, kv_capacity // 1024)
    num_long = min(num_long, 24)
    for _ in range(num_long):
      length = randint(768, 1536)
      prompts.append([randint(0, 10000) for _ in range(length)])
      params.append(
        SamplingParams(
          temperature=0.6,
          ignore_eos=True,
          max_tokens=16,
        )
      )

    # 短 prompt + 较长输出：制造 decode 压力
    num_short = max(64, kv_capacity // 128)
    num_short = min(num_short, 256)
    for _ in range(num_short):
      length = randint(32, 96)
      prompts.append([randint(0, 10000) for _ in range(length)])
      params.append(
        SamplingParams(
          temperature=0.6,
          ignore_eos=True,
          max_tokens=randint(128, 256),
        )
      )

    # 打乱请求，避免单一模式先跑完
    paired = list(zip(prompts, params))
    shuffle(paired)
    prompts = [p for p, _ in paired]
    params = [sp for _, sp in paired]

    num_seqs = len(prompts)
    total_input = sum(len(p) for p in prompts)
    total_max_output = sum(sp.max_tokens for sp in params)
    logger.info(
      f"Chunked mixed pressure test: {num_seqs} seqs "
      f"({num_long} long + {num_short} short), "
      f"total_input={total_input}, total_max_output={total_max_output}, "
      f"peak={total_input + total_max_output}, kv_capacity={kv_capacity}"
    )

    t = time.time()
    results = engine.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - t

    assert len(results) == num_seqs
    for i, result in enumerate(results):
      expected_len = params[i].max_tokens
      assert len(result["token_ids"]) == expected_len, (
        f"Request {i}: expected {expected_len} tokens, " f"got {len(result['token_ids'])}"
      )

    total_output = sum(len(r["token_ids"]) for r in results)
    logger.info(
      f"Chunked mixed pressure PASSED: "
      f"{total_output} tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 6: 极端场景 — 请求数 >> max_num_seqs
# ===================================================================
class TestExtremeRequestCount:
  """
  请求数远超 max_num_seqs，验证调度器的
  waiting_queue 排队机制正常工作。
  """

  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=False,
    )
    _warmup(llm)
    yield llm

  def test_request_count_far_exceeds_batch_limit(self, engine):
    """
    512 条请求 (超过 max_num_seqs=256)，
    每条短 prompt + 短输出。
    应全部完成，不 OOM。
    """
    seed(999)
    num_seqs = 512
    prompt_len = 256 - 2
    output_tokens = 2

    prompt_token_ids = [[randint(0, 10000) for _ in range(prompt_len)] for _ in range(num_seqs)]
    sampling_params = [
      SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_tokens)
      for _ in range(num_seqs)
    ]

    logger.info(
      f"Extreme request count test: {num_seqs} seqs, "
      f"prompt_len={prompt_len}, output_tokens={output_tokens}"
    )

    t = time.time()
    results = engine.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    elapsed = time.time() - t

    assert len(results) == num_seqs
    for i, result in enumerate(results):
      assert len(result["token_ids"]) == output_tokens

    total_output = num_seqs * output_tokens
    logger.info(
      f"Extreme request count PASSED: "
      f"{total_output} tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 7: 真实问题测试
# ===================================================================
class TestRealWorldQuestionsSmoke:
  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=False,
    )
    _warmup(llm)
    yield llm

  def test_real_world_questions_smoke(self, engine):
    """
    小规模真实问题并发生成。
    验证引擎在自然语言实际场景下可稳定完成请求。
    """
    # 扩展为 32 条请求，覆盖并发调度路径。
    prompts = REAL_WORLD_QUESTIONS * 4
    output_tokens = 64
    params = [
      SamplingParams(
        temperature=0.2,
        ignore_eos=True,
        max_tokens=output_tokens,
      )
      for _ in range(len(prompts))
    ]

    logger.info(f"Real-world question smoke test: {len(prompts)} seqs")
    t = time.time()
    results = engine.generate(prompts, params, use_tqdm=True)
    elapsed = time.time() - t

    assert len(results) == len(prompts)
    for i, result in enumerate(results):
      assert len(result["token_ids"]) == output_tokens, (
        f"Request {i}: expected {output_tokens} tokens, " f"got {len(result['token_ids'])}"
      )
      assert result["text"].strip(), f"Request {i}: empty output text"

    total_output = sum(len(r["token_ids"]) for r in results)
    logger.info(
      f"Real-world question smoke test PASSED: "
      f"{total_output} tokens in {elapsed:.2f}s "
      f"({total_output / elapsed:.1f} tok/s)"
    )


# ===================================================================
# Test 8: 真实问题逐 token 对比 transformers
# ===================================================================
class TestRealWorldVsTransformers:
  @pytest.fixture(scope="class")
  def engine(self):
    llm = LLMEngine(
      model=MODEL_NAME,
      enable_chunked_prefill=False,
    )
    _warmup(llm)
    yield llm

  def test_real_question_token_precision_vs_transformers(self, engine):
    """
    对单个真实问题做逐 token greedy 对齐：
    - 打印每步 miniinfer 与 transformers 选中的 token
    - 统计 token-level 匹配率
    """
    try:
      from transformers import AutoModelForCausalLM
    except Exception as exc:
      pytest.skip(f"transformers not available: {exc}")

    max_new_tokens = int(os.getenv("MINIINFER_HF_COMPARE_MAX_NEW_TOKENS", "32"))
    min_match_rate = float(os.getenv("MINIINFER_HF_MIN_MATCH_RATE", "1.0"))
    hf_device = os.getenv("MINIINFER_HF_COMPARE_DEVICE", "cpu").lower()
    if hf_device not in ("cpu", "cuda"):
      pytest.skip(f"Invalid MINIINFER_HF_COMPARE_DEVICE={hf_device}")
    if hf_device == "cuda" and not torch.cuda.is_available():
      pytest.skip("MINIINFER_HF_COMPARE_DEVICE=cuda but CUDA is unavailable.")

    prompt = REAL_WORLD_QUESTIONS[0]
    prompt_token_ids = engine._encode_prompt(prompt)
    sampling_params = SamplingParams(
      temperature=0.0,
      top_k=0,
      top_p=1.0,
      ignore_eos=True,
      max_tokens=max_new_tokens,
    )

    miniinfer_token_ids = []
    for output in engine.stream_generate(prompt_token_ids, sampling_params):
      miniinfer_token_ids.append(int(output.token_id))
      if output.finished:
        break

    assert len(miniinfer_token_ids) == max_new_tokens, (
      f"Expected {max_new_tokens} MiniInfer output tokens, " f"got {len(miniinfer_token_ids)}"
    )

    hf_model = None
    try:
      model_kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
      )
      if hf_device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
      hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except Exception as exc:
      pytest.skip(
        "Cannot load local HF model for comparison. "
        f"Please pre-download model '{MODEL_NAME}'. Error: {exc}"
      )

    hf_model = hf_model.to(hf_device)
    hf_model.eval()
    hf_token_ids = _collect_hf_greedy_tokens(
      hf_model=hf_model,
      prompt_token_ids=prompt_token_ids,
      max_new_tokens=max_new_tokens,
      device=hf_device,
    )

    assert len(hf_token_ids) == max_new_tokens

    print("\n" + "=" * 80)
    print("MiniInfer vs Transformers (token-by-token)")
    print(f"Prompt: {prompt}")
    print("=" * 80)

    match_count = 0
    for step_idx, (tid_mi, tid_hf) in enumerate(zip(miniinfer_token_ids, hf_token_ids), start=1):
      piece_mi = _token_piece(engine.tokenizer, tid_mi)
      txt_mi = _decode_token(engine.tokenizer, tid_mi)
      piece_hf = _token_piece(engine.tokenizer, tid_hf)
      txt_hf = _decode_token(engine.tokenizer, tid_hf)
      matched = tid_mi == tid_hf
      if matched:
        match_count += 1
      print(
        f"[step {step_idx:02d}] "
        f"MiniInfer: id={tid_mi:>6} piece={piece_mi!r} text={txt_mi!r} | "
        f"Transformers: id={tid_hf:>6} piece={piece_hf!r} text={txt_hf!r} | "
        f"match={matched}"
      )

    match_rate = match_count / max_new_tokens
    print("-" * 80)
    print(
      f"Token match: {match_count}/{max_new_tokens} "
      f"({match_rate:.2%}), threshold={min_match_rate:.2%}"
    )
    print(
      "MiniInfer completion:",
      repr(engine.tokenizer.decode(miniinfer_token_ids, skip_special_tokens=False)),
    )
    print(
      "Transformers completion:",
      repr(engine.tokenizer.decode(hf_token_ids, skip_special_tokens=False)),
    )
    print("=" * 80)

    assert match_rate >= min_match_rate, (
      f"Token match rate {match_rate:.2%} is below threshold " f"{min_match_rate:.2%}"
    )

    del hf_model
    gc.collect()
    if hf_device == "cuda":
      torch.cuda.empty_cache()


# ===================================================================
# 独立运行入口
# ===================================================================
if __name__ == "__main__":
  """直接运行此脚本以快速验证"""
  print("=" * 60)
  print("Memory Pressure Stress Tests")
  print("=" * 60)

  # # 创建一个共享引擎用于非 chunked 测试
  llm = LLMEngine(
    model=MODEL_NAME,
    enable_chunked_prefill=False,
  )
  # _warmup(llm)
  # kv_cap = _get_kv_cache_capacity(llm)
  # print(f"\nKV cache capacity: {kv_cap} tokens")

  # --- Test 1: Prefill 压力 ---
  # print("\n" + "-" * 60)
  # print("Test 1: Prefill exceeds KV cache")
  # print("-" * 60)
  # test_prefill = TestPrefillExceedsMemory()
  # test_prefill.test_many_long_prefills(llm)

  # # --- Test 2: Decode 压力 ---
  # print("\n" + "-" * 60)
  # print("Test 2: Decode exceeds KV cache")
  # print("-" * 60)
  # test_decode = TestDecodeExceedsMemory()
  # test_decode.test_many_concurrent_decodes(llm)

  # # --- Test 3: 混合压力 ---
  # print("\n" + "-" * 60)
  # print("Test 3: Mixed pressure")
  # print("-" * 60)
  # test_mixed = TestMixedPressure()
  # test_mixed.test_mixed_long_prefill_and_decode(llm)

  # # --- Test 4: Chunked Prefill ---
  # print("\n" + "-" * 60)
  # print("Test 4: Chunked Prefill")
  # print("-" * 60)
  # # 先释放之前的 engine，避免两个模型同时占用显存导致 OOM
  # llm.stop()
  # del llm
  # gc.collect()
  # torch.cuda.empty_cache()

  # llm_chunked = LLMEngine(
  #     model=MODEL_NAME,
  #     enable_chunked_prefill=True,
  #     chunked_prefill_size=2048,
  # )
  # _warmup(llm_chunked)
  # test_chunked = TestChunkedPrefillExceedsMemory()
  # test_chunked.test_single_very_long_prefill(llm_chunked)
  # test_chunked.test_multiple_long_prefills_chunked(llm_chunked)

  # --- Test 5: 极端请求数 ---
  print("\n" + "-" * 60)
  print("Test 5: Extreme request count")
  print("-" * 60)
  # # 释放 chunked engine，创建新的 engine 用于 Test 5
  # llm_chunked.stop()
  # del llm_chunked
  gc.collect()
  torch.cuda.empty_cache()

  llm = LLMEngine(
    model=MODEL_NAME,
    enable_chunked_prefill=False,
  )
  test_compare = TestRealWorldVsTransformers()
  test_compare.test_real_question_token_precision_vs_transformers(llm)
  # test_real = TestRealWorldQuestionsSmoke()
  # test_real.test_real_world_questions_smoke(llm)
  # _warmup(llm)
  # test_extreme = TestExtremeRequestCount()
  # test_extreme.test_request_count_far_exceeds_batch_limit(llm)

  # print("\n" + "=" * 60)
  # print("ALL MEMORY PRESSURE TESTS PASSED")
  # print("=" * 60)
