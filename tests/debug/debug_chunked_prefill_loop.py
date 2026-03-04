"""
Debug: Chunked Prefill 无限循环问题

问题描述:
从日志可以看到 PrefillAdder 不断初始化但请求没有向前推进：
- available=22 (KV cache 可用 token 数)
- extend_input_len=21 (请求需要的 token 数)
- 请求一直在循环，无法被调度

根本原因分析:
在 prefill_adder.py 的 try_add_prefill 方法中：
1. total_tokens = extend_len + decode_reserve (21 + 1~N = 22+)
2. 检查: if total_tokens >= rem_total_tokens (22 >= 22)
3. 返回 NO_TOKEN，请求被放回 waiting_queue
4. 下一轮 schedule() 调用再次尝试，重复循环

解决方案:
1. 增大 KV cache 容量 (根本解决)
2. 修复边界条件判断 (>= 改为 >)
3. 添加进度检测，避免死循环
"""

import logging
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logging.basicConfig(
  level=logging.DEBUG,
  format="%(asctime)s %(levelname)s %(name)s:%(lineno)d - %(message)s",
)

from miniinfer.scheduler.prefill_adder import (
  PrefillAdder,
  AddReqResult,
  CLIP_MAX_NEW_TOKENS,
)

logger = logging.getLogger(__name__)


class MockReq:
  """模拟请求对象"""

  def __init__(
    self,
    req_id: int,
    extend_input_len: int,
    max_tokens: int = 100,
    output_ids: list = None,
  ):
    self.req_id = req_id
    self.extend_input_len = extend_input_len
    self.max_tokens = max_tokens
    self.output_ids = output_ids or []
    self.origin_input_ids = list(range(extend_input_len))
    self.fill_ids = self.origin_input_ids.copy()
    self.prefix_indices = []
    self.cache_protected_len = 0


def analyze_budget(
  available_kv_tokens: int,
  extend_input_len: int,
  max_tokens: int = 100,
  page_size: int = 256,
  new_token_ratio: float = 0.3,
):
  """分析预算是否足够"""
  print("=" * 60)
  print("预算分析")
  print("=" * 60)
  print(f"  available_kv_tokens: {available_kv_tokens}")
  print(f"  extend_input_len:    {extend_input_len}")
  print(f"  max_tokens:          {max_tokens}")
  print(f"  page_size:           {page_size}")
  print(f"  new_token_ratio:     {new_token_ratio}")
  print()

  # 模拟 try_add_prefill 的计算
  extend_len = extend_input_len
  output_len = 0  # 新请求

  # 对齐到 page_size
  aligned_extend = ((extend_len + page_size - 1) // page_size) * page_size
  print(f"  aligned_extend (ceil to page_size): {aligned_extend}")

  # 计算 decode 预留
  decode_reserve = min(max(max_tokens - output_len, 0), CLIP_MAX_NEW_TOKENS)
  print(f"  decode_reserve (clipped): {decode_reserve}")

  # total_tokens = extend + decode_reserve
  total_tokens = extend_len + decode_reserve
  print(f"  total_tokens (extend + decode): {total_tokens}")

  # 创建 PrefillAdder
  adder = PrefillAdder(
    available_kv_tokens=available_kv_tokens,
    running_reqs=[],  # 没有正在运行的请求
    new_token_ratio=new_token_ratio,
    max_extend_len=8192,
    max_batch_size=256,
    chunk_size=2048,
    page_size=page_size,
  )

  print()
  print("PrefillAdder 状态:")
  print(f"  rem_total_tokens:  {adder.rem_total_tokens}")
  print(f"  cur_rem_tokens:    {adder.cur_rem_tokens}")
  print(f"  rem_input_tokens:  {adder.rem_input_tokens}")
  print(f"  rem_chunk_tokens:  {adder.rem_chunk_tokens}")

  print()
  print("条件检查 (try_add_prefill):")

  # 检查 1: batch size
  print(f"  1. batch size check: {adder.total_reqs} < {adder.max_batch_size} => PASS")

  # 检查 2: total tokens (关键问题)
  check2 = total_tokens >= adder.rem_total_tokens
  print(
    f"  2. total_tokens >= rem_total_tokens: {total_tokens} >= {adder.rem_total_tokens} => {check2}"
  )
  if check2:
    print(f"     !!! 这就是问题所在: 返回 NO_TOKEN, 请求被放回队列 !!!")

  # 检查 3: input tokens
  input_tokens = aligned_extend
  check3 = input_tokens >= adder.rem_input_tokens and len(adder.can_run_list) > 0
  print(
    f"  3. input_tokens >= rem_input_tokens (with can_run_list > 0): {input_tokens} >= {adder.rem_input_tokens} and {len(adder.can_run_list)} > 0 => {check3}"
  )

  print()
  print("诊断结果:")
  if total_tokens > available_kv_tokens:
    print(f"  ERROR: KV cache 容量 ({available_kv_tokens}) 太小!")
    print(f"  需要至少: {total_tokens} tokens (extend={extend_len} + decode={decode_reserve})")
    min_required = extend_len + 1  # 至少要能容纳 extend + 1 个 decode token
    print(f"  最低要求: {min_required} tokens")
  elif total_tokens == available_kv_tokens:
    print(f"  WARNING: 边界条件问题!")
    print(f"  当 total_tokens == rem_total_tokens 时:")
    print(f"    - 当前代码: >= 判断返回 NO_TOKEN (无法调度)")
    print(f"    - 建议修复: 改为 > 判断 (允许边界情况)")
  else:
    print(f"  OK: 预算充足")

  # 尝试添加请求
  print()
  print("=" * 60)
  print("尝试添加请求")
  print("=" * 60)

  req = MockReq(req_id=1, extend_input_len=extend_input_len, max_tokens=max_tokens)
  result = adder.try_add_prefill(req)
  print(f"  Result: {result}")
  print(f"  can_run_list: {len(adder.can_run_list)} requests")

  return result


def simulate_scheduling_loop(
  available_kv_tokens: int,
  extend_input_len: int,
  max_iterations: int = 10,
):
  """模拟调度循环，检测死循环"""
  print()
  print("=" * 60)
  print("模拟调度循环")
  print("=" * 60)

  req = MockReq(req_id=1, extend_input_len=extend_input_len)
  waiting_queue = [req]
  iterations = 0
  consecutive_no_token = 0

  while waiting_queue and iterations < max_iterations:
    iterations += 1
    print(f"\n--- 第 {iterations} 轮调度 ---")

    adder = PrefillAdder(
      available_kv_tokens=available_kv_tokens,
      running_reqs=[],
      new_token_ratio=0.3,
      max_extend_len=8192,
      max_batch_size=256,
      chunk_size=2048,
      page_size=256,
    )

    remaining_waiting = []
    scheduled = False

    for req in waiting_queue:
      if adder.no_remaining_budget():
        remaining_waiting.append(req)
        print(f"  req {req.req_id}: 预算不足，放回队列")
        continue

      result = adder.try_add_prefill(req)
      print(f"  req {req.req_id}: try_add_prefill => {result}")

      if result == AddReqResult.NO_TOKEN:
        remaining_waiting.append(req)
        consecutive_no_token += 1
      elif result == AddReqResult.OTHER:
        remaining_waiting.append(req)
      else:
        scheduled = True
        consecutive_no_token = 0

    waiting_queue = remaining_waiting

    if consecutive_no_token >= 3:
      print(f"\n!!! 检测到死循环: 连续 {consecutive_no_token} 次 NO_TOKEN !!!")
      print("原因: KV cache 容量不足，无法调度任何请求")
      break

  if iterations >= max_iterations:
    print(f"\n!!! 达到最大迭代次数 {max_iterations}，可能存在死循环 !!!")

  return consecutive_no_token >= 3


def suggest_fix():
  """建议修复方案"""
  print()
  print("=" * 60)
  print("修复建议 (已实施)")
  print("=" * 60)
  print(
    """
问题代码 (prefill_adder.py:try_add_prefill):
    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN

问题:
    1. 使用 >= 导致边界条件无法通过
    2. 没有考虑 chunked prefill 可以分步处理请求
    3. 死循环时没有检测和恢复机制

已实施的修复:

修复 1: prefill_adder.py 中的 try_add_prefill
    - 当 total_tokens > rem_total_tokens 但启用了 chunked prefill 时
    - 尝试截断请求为可用的 chunk 大小而不是直接返回 NO_TOKEN
    - 这样即使预算紧张也能逐步推进

修复 2: llm_engine.py 中的 _calc_max_total_tokens
    - 添加最小 KV cache 保证: min_kv_tokens = 2 * page_size
    - 当 KV cache 容量过小时发出警告
    - 在日志中添加更多调试信息 (raw_tokens, min_kv_tokens)

修复 3: scheduler.py 中添加死循环检测
    - 跟踪连续无进展的轮数 (prefill_no_progress_ticks)
    - 当超过阈值时打印警告日志
    - 帮助快速定位类似问题
"""
  )


def diagnose_kvcache_size():
  """诊断 KV cache 容量为什么只有 22 tokens"""
  print()
  print("=" * 60)
  print("诊断: 为什么 available_tokens 只有 22?")
  print("=" * 60)

  # 模拟 _calc_max_total_tokens 的计算过程
  print("\n模拟 _calc_max_total_tokens 计算:")
  print("-" * 40)

  # 假设参数 (需要根据实际模型调整)
  page_size = 256
  reserved_ratio = 0.20

  # 场景1: 可用显存极小
  print("\n场景 1: 可用显存极小时的计算")
  for available_gb in [0.01, 0.1, 0.5, 1.0]:
    available_memory = int(available_gb * 1024**3)
    usable_memory = int(available_memory * (1.0 - reserved_ratio))

    # 假设 bytes_per_token (Qwen2-0.5B 级别)
    # 2 (K+V) * 24 layers * 2 heads * 64 head_dim * 2 bytes = 12,288 bytes
    bytes_per_token = 2 * 24 * 2 * 64 * 2  # 12288

    raw_tokens = usable_memory // bytes_per_token
    tokens_minus_page = raw_tokens - page_size
    aligned_tokens = (tokens_minus_page // page_size) * page_size
    final_tokens = max(aligned_tokens, page_size)

    print(
      f"  available={available_gb}GB: "
      f"raw={raw_tokens}, minus_page={tokens_minus_page}, "
      f"aligned={aligned_tokens}, final={final_tokens}"
    )

  # 场景2: page_size 对齐问题
  print("\n场景 2: page_size 对齐导致容量大幅减少")
  print("  当 raw_tokens < 2*page_size 时:")
  print("    tokens_minus_page = raw_tokens - page_size < page_size")
  print("    aligned = (tokens_minus_page // page_size) * page_size = 0")
  print("    final = max(0, page_size) = page_size")
  print()
  print("  !!! 但日志显示 available=22，不是 256 !!!")
  print("  这说明问题不在 _calc_max_total_tokens，而是:")
  print("    1. KVCacheManager 初始化时 size 参数就很小")
  print("    2. 或者 token_allocator 已经分配了大部分空间")

  # 分析 22 这个数字
  print("\n分析 available=22 的来源:")
  print("-" * 40)
  print("  KVCacheManager.available_tokens() = ")
  print("    token_allocator.available_size() + prefix_cache.evictable_size()")
  print()
  print("  22 这个数字可能来自:")
  print("    - KVCacheManager 初始化时 size=22+N (N 被其他请求占用)")
  print("    - 或者 size 本身就是一个很小的数，然后减去 page_size 对齐损耗")

  print("\n建议添加到 LLMEngine._calc_max_total_tokens 的调试代码:")
  print("-" * 40)
  print(
    """
    logger.info(
        f"DEBUG _calc_max_total_tokens: "
        f"available_memory={available_memory}, "
        f"usable_memory={usable_memory}, "
        f"bytes_per_token={bytes_per_token}, "
        f"raw_tokens={usable_memory // bytes_per_token}, "
        f"page_size={page_size}"
    )
    """
  )


def analyze_page_size_issue():
  """分析 page_size=256 带来的问题"""
  print()
  print("=" * 60)
  print("关键发现: page_size=256 导致的问题")
  print("=" * 60)

  print(
    """
问题链路:
1. _calc_max_total_tokens 计算:
   max_total_num_tokens = (usable_memory // bytes_per_token) - page_size
   max_total_num_tokens = (max_total_num_tokens // page_size) * page_size
   max_total_num_tokens = max(max_total_num_tokens, page_size)

2. 如果 GPU 显存紧张:
   - raw_tokens = 300 (举例)
   - tokens_minus_page = 300 - 256 = 44
   - aligned = (44 // 256) * 256 = 0
   - final = max(0, 256) = 256

3. 但日志显示 available=22，说明:
   - size 可能被设为一个很小的值
   - 或者 page_size 配置不一致
   - 或者有其他位置覆盖了计算结果

解决方案:
1. 检查 EngineConfig 中的 page_size 配置
2. 确认 KVCacheManager 初始化时传入的 size 值
3. 添加最小 size 保证: size = max(size, min_kv_tokens)
   其中 min_kv_tokens 应该足够处理最小请求 (如 1024 tokens)
"""
  )


if __name__ == "__main__":
  print("调试 Chunked Prefill 无限循环问题")
  print()

  # 使用日志中的实际数据
  available_kv = 22  # 从日志: available=22
  extend_len = 21  # 从日志: extend_input_len=21

  # 分析预算
  analyze_budget(
    available_kv_tokens=available_kv,
    extend_input_len=extend_len,
    max_tokens=100,  # 假设的 max_tokens
    page_size=256,  # 从日志: 默认 page_size
  )

  # 模拟调度循环
  is_deadlock = simulate_scheduling_loop(
    available_kv_tokens=available_kv,
    extend_input_len=extend_len,
  )

  # 建议修复
  suggest_fix()

  # 诊断 KV cache 容量
  diagnose_kvcache_size()

  # 分析 page_size 问题
  analyze_page_size_issue()
