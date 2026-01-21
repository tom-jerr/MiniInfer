"""
增量流式解码器测试

测试策略：
1. 使用多种文本（中文、英文、混合、特殊字符）作为 ground truth
2. 用 tokenizer 将文本转换为 token ids
3. 模拟流式解码过程，逐个 token 调用 decode 方法
4. 验证最终解码结果与原始文本一致
"""


import pytest

from typing import List, Tuple
from transformers import AutoTokenizer

import sys
sys.path.insert(0, "/MiniInfer-ws/MiniInfer")

from engine.detokenizer import IncrementalDecoder, DecodeState


# 仅在 pytest 可用时定义测试类
class TestIncrementalDecoder:

    @pytest.fixture
    def tokenizer(self):
        """加载 tokenizer"""
        # 使用 Qwen2 tokenizer，支持中英文
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2-0.5B",
                trust_remote_code=True,
            )
        except Exception:
            # 如果无法加载 Qwen，使用 GPT2 作为后备
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return tokenizer
    
    @pytest.fixture
    def decoder(self, tokenizer):
        """创建解码器实例"""
        return IncrementalDecoder(
            tokenizer=tokenizer,
            skip_special_tokens=True,
        )
    
    # ==================== 基础功能测试 ====================
    
    def test_simple_english_text(self, tokenizer, decoder):
        """测试简单英文文本"""
        ground_truth = "Hello, world! This is a test."
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=1)
    
    def test_simple_chinese_text(self, tokenizer, decoder):
        """测试简单中文文本"""
        ground_truth = "你好，世界！这是一个测试。"
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=2)
    
    def test_mixed_language_text(self, tokenizer, decoder):
        """测试中英文混合文本"""
        ground_truth = "Hello你好，world世界！This is 一个 test测试。"
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=3)
    
    def test_long_text(self, tokenizer, decoder):
        """测试长文本"""
        ground_truth = """
        人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        
        The quick brown fox jumps over the lazy dog.
        Pack my box with five dozen liquor jugs.
        """
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=4)
    
    def test_special_characters(self, tokenizer, decoder):
        """测试特殊字符"""
        ground_truth = "Special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=5)
    
    def test_unicode_emoji(self, tokenizer, decoder):
        """测试 Unicode emoji"""
        ground_truth = "Hello 👋 World 🌍! Python 🐍 is awesome 🚀"
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=6)
    
    def test_numbers_and_math(self, tokenizer, decoder):
        """测试数字和数学符号"""
        ground_truth = "计算: 1 + 2 = 3, 10 * 20 = 200, π ≈ 3.14159"
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=7)
    
    def test_code_snippet(self, tokenizer, decoder):
        """测试代码片段"""
        ground_truth = '''def hello():
    print("Hello, World!")
    return 42'''
        self._verify_incremental_decode(tokenizer, decoder, ground_truth, req_id=8)
    
    # ==================== 批量解码测试 ====================
    
    def test_batch_decode(self, tokenizer, decoder):
        """测试批量解码"""
        texts = [
            "Hello, world!",
            "你好，世界！",
            "Mixed 混合 text 文本",
        ]
        
        # 编码所有文本
        all_token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
        max_len = max(len(ids) for ids in all_token_ids)
        
        # 模拟批量解码
        results = [""] * len(texts)
        for step in range(max_len):
            req_ids = []
            token_ids = []
            indices = []
            
            for i, ids in enumerate(all_token_ids):
                if step < len(ids):
                    req_ids.append(100 + i)  # 使用不同的 req_id
                    token_ids.append(ids[step])
                    indices.append(i)
            
            batch_results = decoder.decode_batch(req_ids, token_ids)
            for idx, (delta, _) in zip(indices, batch_results):
                results[idx] += delta
        
        # 刷新剩余文本
        for i in range(len(texts)):
            remaining = decoder.flush(100 + i)
            results[i] += remaining
        
        # 验证
        for i, (result, ground_truth) in enumerate(zip(results, texts)):
            assert result == ground_truth, f"Text {i} mismatch: '{result}' != '{ground_truth}'"
    
    # ==================== EOS 处理测试 ====================
    
    def test_eos_handling(self, tokenizer, decoder):
        """测试 EOS token 处理"""
        ground_truth = "Test EOS handling"
        token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
        eos_token_id = tokenizer.eos_token_id
        
        # 添加 EOS token
        token_ids.append(eos_token_id)
        
        result = ""
        finished = False
        for token_id in token_ids:
            delta, finished = decoder.decode(200, token_id, eos_token_id)
            result += delta
            if finished:
                break
        
        assert finished, "Should be finished after EOS"
        assert result == ground_truth, f"Mismatch: '{result}' != '{ground_truth}'"
    
    # ==================== 状态管理测试 ====================
    
    def test_cleanup(self, tokenizer, decoder):
        """测试状态清理"""
        ground_truth = "Test cleanup"
        token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
        
        # 解码
        for token_id in token_ids:
            decoder.decode(300, token_id)
        
        # 验证状态存在
        assert 300 in decoder._states
        
        # 清理
        decoder.cleanup(300)
        assert 300 not in decoder._states
    
    def test_multiple_requests(self, tokenizer, decoder):
        """测试多个并发请求"""
        texts = {
            400: "First request text",
            401: "Second request 第二个请求",
            402: "Third 第三 request 请求",
        }
        
        # 编码
        token_ids_map = {
            req_id: tokenizer.encode(text, add_special_tokens=False)
            for req_id, text in texts.items()
        }
        
        # 交错解码
        max_len = max(len(ids) for ids in token_ids_map.values())
        results = {req_id: "" for req_id in texts}
        
        for step in range(max_len):
            for req_id, token_ids in token_ids_map.items():
                if step < len(token_ids):
                    delta, _ = decoder.decode(req_id, token_ids[step])
                    results[req_id] += delta
        
        # 刷新并验证
        for req_id, ground_truth in texts.items():
            results[req_id] += decoder.flush(req_id)
            assert results[req_id] == ground_truth, \
                f"Request {req_id} mismatch: '{results[req_id]}' != '{ground_truth}'"
    
    def test_get_stats(self, tokenizer, decoder):
        """测试统计信息"""
        # 解码一些 token
        text = "Test stats"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        for token_id in token_ids:
            decoder.decode(500, token_id)
        
        stats = decoder.get_stats()
        assert stats["active_requests"] == 1
        assert stats["total_tokens_decoded"] == len(token_ids)
    
    # ==================== 辅助方法 ====================
    
    def _verify_incremental_decode(
        self,
        tokenizer,
        decoder: IncrementalDecoder,
        ground_truth: str,
        req_id: int,
    ):
        """
        验证增量解码的正确性
        
        Args:
            tokenizer: 分词器
            decoder: 增量解码器
            ground_truth: 原始文本（作为 ground truth）
            req_id: 请求 ID
        """
        # 1. 编码文本
        token_ids = tokenizer.encode(ground_truth, add_special_tokens=False)
        
        # 2. 模拟流式解码，收集增量文本
        collected_deltas: List[str] = []
        for token_id in token_ids:
            delta, finished = decoder.decode(req_id, token_id)
            if delta:
                collected_deltas.append(delta)
        
        # 3. 刷新剩余文本
        remaining = decoder.flush(req_id)
        if remaining:
            collected_deltas.append(remaining)
        
        # 4. 拼接增量文本
        result = "".join(collected_deltas)
        
        # 5. 验证结果
        assert result == ground_truth, \
            f"Incremental decode mismatch:\n" \
            f"  Expected: '{ground_truth}'\n" \
            f"  Got:      '{result}'\n" \
            f"  Token IDs: {token_ids}\n" \
            f"  Deltas:    {collected_deltas}"
        
        # 6. 验证 get_full_text 方法
        full_text = decoder.get_full_text(req_id)
        assert full_text == ground_truth, \
            f"get_full_text mismatch: '{full_text}' != '{ground_truth}'"
        
        # 7. 清理状态
        decoder.cleanup(req_id)



class TestIncrementalDecoderWithCallback:
    """测试带回调的增量解码器"""
    
    @pytest.fixture
    def tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
        except Exception:
            return AutoTokenizer.from_pretrained("gpt2")
    
    def test_callback_invocation(self, tokenizer):
        """测试回调函数调用"""
        callback_records = []
        
        def callback(req_id, token_id, delta_text, finished):
            callback_records.append({
                "req_id": req_id,
                "token_id": token_id,
                "delta_text": delta_text,
                "finished": finished,
            })
        
        decoder = IncrementalDecoder(
            tokenizer=tokenizer,
            on_token_callback=callback,
        )
        
        text = "Hello, world!"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        for token_id in token_ids:
            decoder.decode(600, token_id)
        
        # 验证回调被调用
        assert len(callback_records) == len(token_ids)
        assert all(r["req_id"] == 600 for r in callback_records)
        assert all(r["finished"] is False for r in callback_records)
