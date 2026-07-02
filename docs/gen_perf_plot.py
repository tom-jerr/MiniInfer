"""生成 MiniInfer vs vLLM 性能对比柱状图。

使用本会话中收集的 benchmark 结果（FA2, Qwen3-0.6B, A100-80GB）。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, MiniInfer 初版, MiniInfer 优化版, vLLM)
# 数据来自本会话实测（FA2, block_size=256, gpu_memory_utilization=0.9）
DATA = [
    ("256×1024\nbalanced",      6422,  8420,  10136),
    ("128×1024\ndecode-heavy",  7233, 14053, 14359),
    ("256×1024\ndecode-heavy",  None, 17369, 16333),  # 初版 KV 不足无法跑
    ("256×256\nshort seq",      None, 10552, 10050),
]

labels = [d[0] for d in DATA]
mi_init = [d[1] if d[1] else 0 for d in DATA]
mi_opt  = [d[2] for d in DATA]
vllm    = [d[3] for d in DATA]

x = np.arange(len(labels))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w, mi_init, w, label="MiniInfer (initial)", color="#90CAF9", edgecolor="white")
b2 = ax.bar(x,     mi_opt,  w, label="MiniInfer (optimized)", color="#1565C0", edgecolor="white")
b3 = ax.bar(x + w, vllm,    w, label="vLLM (FA2)", color="#FF7043", edgecolor="white")

ax.set_ylabel("Throughput (tok/s)", fontsize=12)
ax.set_title("MiniInfer vs vLLM — Qwen3-0.6B, FA2, A100-80GB", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

# 标注数值
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 80, f"{int(h)}",
                    ha="center", va="bottom", fontsize=8)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "img", "perf_comparison.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
