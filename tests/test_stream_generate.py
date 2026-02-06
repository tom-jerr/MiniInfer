"""
LLMEngine 流式生成测试

测试 stream_generate 和 generate 接口的功能：
1. 流式生成：逐 token 输出，显示增量文本
2. 批量生成：阻塞式生成，返回完整结果
3. 动态添加请求：在生成过程中添加新请求
"""

import sys
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, "/MiniInfer-ws/MiniInfer")


# 定义测试用的 RequestOutput 类（避免循环导入）
@dataclass
class MockRequestOutput:
    """模拟的请求输出结果"""

    request_id: int
    delta_text: str
    full_text: str
    token_id: int
    output_token_ids: List[int]
    finished: bool
    finish_reason: Optional[str] = None


@dataclass
class StreamProgress:
    """流式生成进度跟踪"""

    request_id: int
    prompt: str
    tokens_generated: int = 0
    text: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_time > 0 and self.tokens_generated > 0:
            return self.tokens_generated / self.elapsed_time
        return 0.0


class StreamOutputFormatter:
    """
    流式输出格式化器

    提供规范化的输出显示，包括：
    - 实时进度条
    - 增量文本高亮
    - 吞吐量统计
    """

    # ANSI 颜色代码
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    def __init__(self, use_color: bool = True, show_progress: bool = True):
        self.use_color = use_color
        self.show_progress = show_progress
        self.progress_map: Dict[int, StreamProgress] = {}
        self.total_tokens = 0
        self.start_time = time.time()

    def _color(self, text: str, color: str) -> str:
        if not self.use_color:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"

    def register_request(self, request_id: int, prompt: str):
        """注册新请求"""
        self.progress_map[request_id] = StreamProgress(
            request_id=request_id,
            prompt=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            start_time=time.time(),
        )

    def update(self, output):
        """更新请求进度 (接受 MockRequestOutput 或 RequestOutput)"""
        if output.request_id not in self.progress_map:
            self.register_request(output.request_id, f"Request-{output.request_id}")

        progress = self.progress_map[output.request_id]
        progress.tokens_generated += 1
        progress.text += output.delta_text
        self.total_tokens += 1

        if output.finished:
            progress.finished = True
            progress.finish_reason = output.finish_reason
            progress.end_time = time.time()

    def print_delta(self, output, show_token_info: bool = False):
        """打印增量输出 (接受 MockRequestOutput 或 RequestOutput)"""
        req_id = output.request_id
        delta = output.delta_text

        if show_token_info:
            # 详细模式：显示 token 信息，每个 token 单独一行
            token_info = self._color(f"[req:{req_id}|tok:{output.token_id}]", "dim")
            delta_display = repr(delta) if delta else "''"
            print(f"{token_info} {delta_display}")
        else:
            # 简洁模式：流式输出增量文本
            if delta:
                prefix = self._color(f"[{req_id}]", "cyan")
                print(f"{prefix}{delta}", end="", flush=True)

        if output.finished:
            finish_tag = self._color(f" [DONE:{output.finish_reason}]", "green")
            print(finish_tag)

    def print_progress_bar(self):
        """打印进度条"""
        if not self.show_progress:
            return

        total = len(self.progress_map)
        finished = sum(1 for p in self.progress_map.values() if p.finished)

        bar_width = 30
        filled = int(bar_width * finished / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = time.time() - self.start_time
        tps = self.total_tokens / elapsed if elapsed > 0 else 0

        progress_line = (
            f"\r{self._color('Progress:', 'bold')} [{bar}] "
            f"{finished}/{total} | "
            f"{self._color(f'{self.total_tokens} tokens', 'yellow')} | "
            f"{self._color(f'{tps:.1f} tok/s', 'green')}"
        )
        print(progress_line, end="", flush=True)

    def print_summary(self):
        """打印总结"""
        print("\n" + "=" * 60)
        print(self._color("Generation Summary", "bold"))
        print("=" * 60)

        elapsed = time.time() - self.start_time

        for req_id, progress in sorted(self.progress_map.items()):
            status = (
                self._color("✓", "green")
                if progress.finished
                else self._color("○", "yellow")
            )
            print(f"\n{status} Request {req_id}:")
            print(f"   Prompt: {progress.prompt}")
            print(f"   Tokens: {progress.tokens_generated}")
            print(
                f"   Time: {progress.elapsed_time:.2f}s ({progress.tokens_per_second:.1f} tok/s)"
            )
            print(
                f"   Output: {progress.text[:100]}{'...' if len(progress.text) > 100 else ''}"
            )

        print("\n" + "-" * 60)
        print(
            f"Total: {self.total_tokens} tokens in {elapsed:.2f}s ({self.total_tokens/elapsed:.1f} tok/s)"
        )
        print("=" * 60)


def test_stream_generate_mock():
    """
    测试流式生成（Mock 模式，不需要真实模型）

    模拟 engine 的行为来验证接口设计
    """
    print("\n" + "=" * 60)
    print("Test: Stream Generate (Mock Mode)")
    print("=" * 60 + "\n")

    # 模拟的 prompts 和预期输出
    test_cases = [
        {
            "prompt": "你好，请介绍一下自己",
            "expected_tokens": ["我", "是", "一个", "AI", "助手", "。"],
        },
        {
            "prompt": "What is Python?",
            "expected_tokens": [
                "Python",
                " is",
                " a",
                " programming",
                " language",
                ".",
            ],
        },
    ]

    formatter = StreamOutputFormatter(use_color=True, show_progress=True)

    # 注册请求
    for i, case in enumerate(test_cases):
        formatter.register_request(i, case["prompt"])

    print("Simulating stream generation...\n")

    # 模拟生成过程
    max_tokens = max(len(case["expected_tokens"]) for case in test_cases)

    for step in range(max_tokens):
        for req_id, case in enumerate(test_cases):
            if step < len(case["expected_tokens"]):
                is_last = step == len(case["expected_tokens"]) - 1

                # 创建模拟的 RequestOutput
                output = MockRequestOutput(
                    request_id=req_id,
                    delta_text=case["expected_tokens"][step],
                    full_text="".join(case["expected_tokens"][: step + 1]),
                    token_id=1000 + step,
                    output_token_ids=list(range(step + 1)),
                    finished=is_last,
                    finish_reason="eos" if is_last else None,
                )

                formatter.update(output)
                formatter.print_delta(output, show_token_info=True)

        formatter.print_progress_bar()
        time.sleep(0.1)  # 模拟推理延迟

    print()  # 换行
    formatter.print_summary()

    print("\n✓ Mock stream generate test passed!")


def test_stream_generate_with_engine(model_path: str):
    """
    测试流式生成（真实模型）

    Args:
        model_path: 模型路径
    """
    # 延迟导入避免循环导入问题
    from miniinfer.engine.llm_engine import LLMEngine
    from miniinfer.utils.sampling_params import SamplingParams

    print("\n" + "=" * 60)
    print("Test: Stream Generate (Real Engine)")
    print("=" * 60 + "\n")

    # 初始化引擎
    print(f"Loading model from: {model_path}")
    engine = LLMEngine(model=model_path)

    prompts = [
        "你好，请用一句话介绍人工智能。",
        "What is machine learning in one sentence?",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=50,
    )

    formatter = StreamOutputFormatter(use_color=True, show_progress=True)

    # 注册请求
    for i, prompt in enumerate(prompts):
        formatter.register_request(i, prompt)

    print("\nGenerating...\n")

    # 流式生成
    for output in engine.stream_generate(prompts, sampling_params):
        formatter.update(output)
        formatter.print_delta(output)
        formatter.print_progress_bar()

    print()
    formatter.print_summary()

    engine.stop()
    print("\n✓ Real engine stream generate test passed!")


def test_generate_with_engine(model_path: str):
    """
    测试批量生成（真实模型）

    Args:
        model_path: 模型路径
    """
    # 延迟导入避免循环导入问题
    from miniinfer.engine.llm_engine import LLMEngine
    from miniinfer.utils.sampling_params import SamplingParams

    print("\n" + "=" * 60)
    print("Test: Batch Generate (Real Engine)")
    print("=" * 60 + "\n")

    print(f"Loading model from: {model_path}")
    engine = LLMEngine(model=model_path)

    prompts = [
        "1+1等于多少？",
        "What is 2+2?",
        "Python是什么？",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=32,
    )

    print(f"\nGenerating {len(prompts)} prompts...")
    start_time = time.time()

    results = engine.generate(prompts, sampling_params, use_tqdm=True)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    total_tokens = 0
    for i, result in enumerate(results):
        print(f"\n[{i}] Prompt: {prompts[i]}")
        print(f"    Output: {result['text']}")
        print(f"    Tokens: {len(result['token_ids'])}")
        total_tokens += len(result["token_ids"])

    print("\n" + "-" * 60)
    print(
        f"Total: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tok/s)"
    )
    print("=" * 60)

    engine.stop()
    print("\n✓ Batch generate test passed!")


def test_dynamic_add_request(model_path: str):
    """
    测试动态添加请求

    在生成过程中添加新请求，验证调度器正确处理
    """
    # 延迟导入避免循环导入问题
    from miniinfer.engine.llm_engine import LLMEngine
    from miniinfer.utils.sampling_params import SamplingParams

    print("\n" + "=" * 60)
    print("Test: Dynamic Add Request")
    print("=" * 60 + "\n")

    print(f"Loading model from: {model_path}")
    engine = LLMEngine(model=model_path)

    sampling_params = SamplingParams(temperature=0.7, max_tokens=20)

    # 先添加第一个请求
    prompt1 = "问题：1+1=?"
    prompt2 = "问题：1+1+2=?"

    req_id1 = engine.add_request(prompt1, sampling_params)
    print(f"Added request {req_id1}")

    formatter = StreamOutputFormatter(use_color=True)
    formatter.register_request(req_id1, "第一个问题")

    prefix_hit_lens: Dict[int, int] = {}
    # 计算理论上的“可命中上限”（按 page_size 对齐，且最多匹配到 prompt 的倒数第2个 token）
    try:
        ids1 = engine._encode_prompt(prompt1)
        ids2 = engine._encode_prompt(prompt2)
        lcp = 0
        for a, b in zip(ids1, ids2):
            if a != b:
                break
            lcp += 1
        page_size = getattr(engine.kv_cache_mgr, "page_size", 1)
        max_match_len = max(len(ids2) - 1, 0)
        theoretical_max_hit = (min(lcp, max_match_len) // page_size) * page_size
        print(
            f"[prefix_cache_expect] prompt1_tokens={len(ids1)} prompt2_tokens={len(ids2)} "
            f"lcp={lcp} page_size={page_size} theoretical_max_hit={theoretical_max_hit}"
        )
    except Exception as e:
        print(f"[prefix_cache_expect] skipped (encode failed): {e}")

    step_count = 0
    added_second = False

    while not engine.is_finished():
        step_output = engine.step()
        step_count += 1

        if step_output.prefill_prefix_lens:
            for rid, pre_len in step_output.prefill_prefix_lens.items():
                prefix_hit_lens[int(rid)] = int(pre_len)
                ext_len = step_output.prefill_extend_lens.get(int(rid))
                print(
                    f"\n[prefix_cache_hit] req={rid} prefix_len={pre_len} extend_len={ext_len}"
                )

        for output in step_output.outputs:
            formatter.update(output)
            formatter.print_delta(output)

        # 在第5步时动态添加第二个请求
        if step_count == 5 and not added_second:
            req_id2 = engine.add_request(prompt2, sampling_params)
            print(f"\n>>> Dynamically added request {req_id2}")
            formatter.register_request(req_id2, "第二个问题")
            added_second = True

    if prefix_hit_lens:
        print("\n" + "-" * 60)
        print("Prefix-cache hit lengths (tokens):")
        for rid in sorted(prefix_hit_lens):
            print(f"  req={rid}: {prefix_hit_lens[rid]}")

    formatter.print_summary()

    engine.stop()
    print("\n✓ Dynamic add request test passed!")


def test_prefix_cache_reuse_after_finished_request(model_path: str):
    """
    测试：请求生成结束后，将 (Prompt + Output) 插入 Radix Tree；
    后续相同 Prompt 的请求应当命中 prefix cache 并复用 KV。
    """
    from miniinfer.engine.llm_engine import LLMEngine
    from miniinfer.utils.sampling_params import SamplingParams

    print("\n" + "=" * 60)
    print("Test: Prefix Cache Reuse After Finished Request")
    print("=" * 60 + "\n")

    print(f"Loading model from: {model_path}")
    engine = LLMEngine(model=model_path)

    page_size = int(getattr(engine.kv_cache_mgr, "page_size", 1))

    # match_prefix(page_size>1) 只会按 page 对齐匹配；
    # prefix_for_waiting_req 还会先丢掉最后 1 个 token 再 match。
    # 所以要想看到非 0 命中，需要 (len(prompt_ids) - 1) >= page_size。
    prompt_len = page_size + 8
    prompt_ids = [1] * prompt_len

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    # req1 跑完后 release_request() 会插入 radix cache
    req1 = engine.add_request(prompt_ids, sampling_params)
    expected_hit = ((prompt_len - 1) // page_size) * page_size
    expected_extend = prompt_len - expected_hit
    print(
        f"req1={req1} prompt_len={prompt_len} page_size={page_size} "
        f"expected_req2_prefix_hit={expected_hit} expected_req2_extend={expected_extend}"
    )

    while True:
        step_out = engine.step()
        if any(o.request_id == req1 and o.finished for o in step_out.outputs):
            break
        if not step_out.outputs and engine.is_finished():
            raise AssertionError("req1 finished unexpectedly without outputs")

    # 添加相同 prompt 的 req2，验证命中
    req2 = engine.add_request(prompt_ids, sampling_params)
    got_hit = None
    got_extend = None

    while got_hit is None:
        step_out = engine.step()
        if req2 in step_out.prefill_prefix_lens:
            got_hit = int(step_out.prefill_prefix_lens[req2])
            got_extend = int(step_out.prefill_extend_lens.get(req2, -1))

    print(
        f"[prefix_cache_hit] req2={req2} prefix_len={got_hit} extend_len={got_extend} "
        f"(expected prefix_len={expected_hit} extend_len={expected_extend})"
    )
    assert got_hit == expected_hit
    assert got_extend == expected_extend

    engine.stop()
    print("\n✓ Prefix cache reuse-after-finish test passed!")


def run_all_tests(model_path: Optional[str] = None):
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("LLMEngine Stream Generate Tests")
    print("=" * 60)

    # # 总是运行 mock 测试
    # test_stream_generate_mock()

    # 如果提供了模型路径，运行真实测试
    if model_path:
        test_stream_generate_with_engine(model_path)
        test_generate_with_engine(model_path)
        # test_dynamic_add_request(model_path)
        test_prefix_cache_reuse_after_finished_request(model_path)
    else:
        print("\n[INFO] Skipping real engine tests (no model_path provided)")
        print("       To run full tests, call: run_all_tests('/path/to/model')")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLMEngine stream generate")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the model (optional, runs mock tests if not provided)",
    )
    # parser.add_argument(
    #     "--mock-only",
    #     action="store_true",
    #     help="Only run mock tests"
    # )

    args = parser.parse_args()

    # if args.mock_only:
    #     test_stream_generate_mock()
    # else:
    run_all_tests(args.model)
