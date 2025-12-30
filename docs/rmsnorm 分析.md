# rmsnorm 分析与优化方案

> rmsnorm 是 Memory Bound 算子

在 Memory Bound 的场景下，性能主要受限于显存带宽。我们通过计算**显存访问次数（Memory Accesses）**的减少比例来估算收益。

假设输入张量形状为 [B, S, H]，元素总数为 N，数据类型为 FP16（2 Bytes）。

## 原始方案 (Add + RMSNorm 分离):

**Add Kernel: x = x + residual**

- 读 x: N
- 读 residual: N
- 写 x: N
- 小计: 3N 次访存
  **RMSNorm Kernel: y = rmsnorm(x)**
- 读 x: N
- 写 y: N
- 小计: 2N 次访存
  总计: 3N+2N=5N 次访存。

## 融合方案 (Fused Add + RMSNorm):

**Fused Kernel:**

- 读 x: N
- 读 residual: N
- (计算 acc = x + residual，保存在寄存器/SRAM中)
- (计算 y = rmsnorm(acc)，保存在寄存器/SRAM中)
- 写 y: N
  总计: 3N 次访存。

## 理论优化结果

**优化比例:**
Bandwidth Savings: $\frac{5N - 3N}{5N}  = \frac{2N}{5N}= 40%$

结论: 显存读写量减少了 40%。

## 优化方案

- Load: 同时加载 x 和 residual 到 SRAM。
- Compute: 在寄存器中完成加法 acc = x + res。
- Reduce: 直接对 acc 进行平方和规约（Row-wise Reduction），计算 RMS 统计量。
- Normalize: 使用统计量对 acc 进行归一化。
- Store: 仅写回最终结果 y (以及可选地写回 acc 到 x，如果后续层需要残差流)。
