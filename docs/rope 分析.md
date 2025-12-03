1. RoPE 的计算逻辑RoPE 的核心是对 Query ($Q$) 和 Key ($K$) 进行旋转。对于一对数值 $(x_1, x_2)$，计算公式为：$$\begin{aligned}
y_1 &= x_1 \cos \theta - x_2 \sin \theta \\
y_2 &= x_1 \sin \theta + x_2 \cos \theta
\end{aligned}$$2. 算术强度估算 (FP16 为例)计算量 (Compute):计算 $y_1$: 2次乘法, 1次减法 = 3 FLOPs计算 $y_2$: 2次乘法, 1次加法 = 3 FLOPs总计：6 FLOPs 处理 2 个元素（即每个元素 3 FLOPs）。访存量 (Memory Access):读取 $x_1, x_2$: 2 × 2 bytes = 4 bytes写入 $y_1, y_2$: 2 × 2 bytes = 4 bytes读取 $\sin, \cos$: 假设缓存命中（L2 Cache），如果没命中还要加 4 bytes。最小访存：8 bytes (只算读写 X/Y)。比率 (Ratio):$$\frac{6 \text{ FLOPs}}{8 \text{ Bytes}} = 0.75 \text{ FLOPs/Byte}$$3. 对比硬件能力 (以 A100 为例)A100 SXM: 算力约 312 TFLOPS (FP16)，带宽约 2000 GB/s。硬件平衡点: $312,000 / 2,000 \approx 156 \text{ FLOPs/Byte}$。结论：$$0.75 \ll 156$$RoPE 的算术强度 远低于 硬件的计算/带宽平衡点。GPU 绝大部分时间都在等待从 HBM 中读取 $Q, K$ 数据和写回结果，计算单元大部分时间是空闲的。因此，RoPE 是一个典型的内存带宽受限 (memory-bound) 算子。

我们未来最好将 RoPE 算子融合到 Attention 或者 MLP 中，以减少内存访问次数，提高整体性能。