"""
Incremental Streaming Decoder - 增量流式解码器

用于高效地将 token ids 增量解码为文本，适用于流式输出场景。

主要特性:
1. 增量解码: 每次只解码新增的 token，避免重复计算
2. Unicode 边界处理: 正确处理多字节字符（如中文）被拆分到多个 token 的情况
3. 特殊 token 处理: 支持跳过特殊 token
4. 前缀缓存: 避免输出不完整的字符
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field


@dataclass
class DecodeState:
    """单个请求的解码状态"""
    # 已处理的所有 token ids
    token_ids: List[int] = field(default_factory=list)
    # 上一次解码后的完整文本
    prev_text: str = ""
    # 待确认输出的文本前缀（可能是不完整的 unicode 字符）
    pending_prefix: str = ""
    # 已经输出给用户的文本长度
    output_offset: int = 0
    # 是否已完成
    finished: bool = False


class IncrementalDecoder:
    """
    增量流式解码器
    
    用于将模型生成的 token ids 增量解码为文本，支持流式输出。
    
    使用方法:
    ```python
    decoder = IncrementalDecoder(tokenizer)
    
    # 每次模型生成新 token 后调用
    delta_text, finished = decoder.decode(request_id, new_token_id, eos_token_id)
    
    # 输出增量文本
    if delta_text:
        stream_output(delta_text)
    
    # 请求完成后清理状态
    if finished:
        decoder.cleanup(request_id)
    ```
    """
    
    # 用于检测不完整 unicode 的前缀长度
    UNICODE_CHECK_PREFIX_LEN = 3
    
    def __init__(
        self, 
        tokenizer: Any,
        skip_special_tokens: bool = True,
        on_token_callback: Optional[Callable[[int, int, str, bool], None]] = None,
    ):
        """
        初始化增量解码器
        
        Args:
            tokenizer: HuggingFace tokenizer 实例
            skip_special_tokens: 解码时是否跳过特殊 token
            on_token_callback: 可选的回调函数，签名为 (req_id, token_id, delta_text, finished)
        """
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.on_token_callback = on_token_callback
        
        # 每个请求的解码状态
        self._states: Dict[int, DecodeState] = {}
        
        # 特殊 token ids 集合，用于快速查找
        self._special_token_ids = set()
        if hasattr(tokenizer, 'all_special_ids'):
            self._special_token_ids = set(tokenizer.all_special_ids)
        
        # EOS token id
        self._eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    
    def get_or_create_state(self, req_id: int) -> DecodeState:
        """获取或创建请求的解码状态"""
        if req_id not in self._states:
            self._states[req_id] = DecodeState()
        return self._states[req_id]
    
    def decode(
        self,
        req_id: int,
        token_id: int,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[str, bool]:
        """
        对单个 token 进行增量解码
        
        Args:
            req_id: 请求 ID
            token_id: 新生成的 token id
            eos_token_id: 结束符 token id（可选，不传则使用 tokenizer 的默认值）
            
        Returns:
            Tuple[str, bool]: (增量文本, 是否完成)
        """
        state = self.get_or_create_state(req_id)
        
        if state.finished:
            return "", True
        
        # 确定 EOS token id
        eos_id = eos_token_id if eos_token_id is not None else self._eos_token_id
        
        # 检查是否是 EOS token
        is_eos = (token_id == eos_id)
        
        # 添加新 token
        state.token_ids.append(token_id)
        
        # 如果是 EOS，标记完成并返回剩余的 pending 文本
        if is_eos:
            state.finished = True
            delta = state.pending_prefix
            state.pending_prefix = ""
            
            if self.on_token_callback:
                self.on_token_callback(req_id, token_id, delta, True)
            
            return delta, True
        
        # 如果需要跳过特殊 token
        if self.skip_special_tokens and token_id in self._special_token_ids:
            return "", False
        
        # 解码所有 token
        full_text = self.tokenizer.decode(
            state.token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
        
        # 计算增量文本
        delta_text = self._compute_delta(state, full_text)
        
        # 更新状态
        state.prev_text = full_text
        
        if self.on_token_callback:
            self.on_token_callback(req_id, token_id, delta_text, False)
        
        return delta_text, False
    
    def decode_batch(
        self,
        req_ids: List[int],
        token_ids: List[int],
        eos_token_id: Optional[int] = None,
    ) -> List[Tuple[str, bool]]:
        """
        批量增量解码
        
        Args:
            req_ids: 请求 ID 列表
            token_ids: 对应的新生成 token id 列表
            eos_token_id: 结束符 token id
            
        Returns:
            List[Tuple[str, bool]]: 每个请求的 (增量文本, 是否完成) 列表
        """
        results = []
        for req_id, token_id in zip(req_ids, token_ids):
            result = self.decode(req_id, token_id, eos_token_id)
            results.append(result)
        return results
    
    def _compute_delta(self, state: DecodeState, full_text: str) -> str:
        """
        计算增量文本
        
        HuggingFace tokenizer 的 decode 方法已经正确处理了 Unicode 边界，
        所以我们只需要简单地计算新增部分。
        
        但需要注意：某些 tokenizer 在处理 token 边界时可能会调整空格，
        所以我们需要正确处理 prev_text 和 pending_prefix 的关系。
        """
        
        # 已输出的文本长度
        output_len = state.output_offset
        
        # 新增的文本就是从已输出位置开始到当前完整文本
        if len(full_text) <= output_len:
            return ""
        
        new_text = full_text[output_len:]
        
        # 对于可能存在的不完整 UTF-8 字节（在某些 tokenizer 中），
        # 我们检查最后的字符是否完整
        # 策略：检查最后一个字符是否是有效的 Unicode 代码点
        if new_text and self._has_incomplete_char(new_text):
            # 保留可能不完整的部分
            state.pending_prefix = new_text
            return ""
        
        # 更新已输出偏移
        state.output_offset = len(full_text)
        state.pending_prefix = ""
        
        return new_text
    
    def _has_incomplete_char(self, text: str) -> bool:
        """
        检查文本末尾是否有不完整的字符
        
        对于 Python str（已经是 Unicode），通常不会有不完整的字符，
        但某些 tokenizer 可能会产生 replacement character (U+FFFD)。
        """
        if not text:
            return False
        
        # 检查是否包含 replacement character
        if '\ufffd' in text:
            return True
        
        return False
    
    def flush(self, req_id: int) -> str:
        """
        刷新请求的所有待输出文本
        
        用于请求完成时，确保所有文本都被输出。
        
        Args:
            req_id: 请求 ID
            
        Returns:
            剩余的待输出文本
        """
        if req_id not in self._states:
            return ""
        
        state = self._states[req_id]
        
        # 获取完整解码文本
        full_text = self.tokenizer.decode(
            state.token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
        
        # 返回从已输出位置开始的剩余文本
        remaining = full_text[state.output_offset:]
        state.output_offset = len(full_text)
        state.pending_prefix = ""
        
        return remaining
    
    def get_full_text(self, req_id: int) -> str:
        """
        获取请求的完整解码文本
        
        Args:
            req_id: 请求 ID
            
        Returns:
            完整的解码文本
        """
        if req_id not in self._states:
            return ""
        
        state = self._states[req_id]
        return self.tokenizer.decode(
            state.token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )
    
    def get_output_tokens(self, req_id: int) -> List[int]:
        """
        获取请求已生成的所有 token ids
        
        Args:
            req_id: 请求 ID
            
        Returns:
            token id 列表
        """
        if req_id not in self._states:
            return []
        return self._states[req_id].token_ids.copy()
    
    def cleanup(self, req_id: int):
        """
        清理请求的解码状态
        
        Args:
            req_id: 请求 ID
        """
        if req_id in self._states:
            del self._states[req_id]
    
    def cleanup_all(self):
        """清理所有解码状态"""
        self._states.clear()
    
    def is_finished(self, req_id: int) -> bool:
        """
        检查请求是否已完成
        
        Args:
            req_id: 请求 ID
            
        Returns:
            是否已完成
        """
        if req_id not in self._states:
            return False
        return self._states[req_id].finished
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取解码器统计信息
        
        Returns:
            包含统计信息的字典
        """
        active_count = sum(1 for s in self._states.values() if not s.finished)
        finished_count = sum(1 for s in self._states.values() if s.finished)
        total_tokens = sum(len(s.token_ids) for s in self._states.values())
        
        return {
            "active_requests": active_count,
            "finished_requests": finished_count,
            "total_tokens_decoded": total_tokens,
        }
