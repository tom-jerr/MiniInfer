# Kernel-Fusion Benchmarks Report

## Baseline
使用 Pytorch 原生 Attention 和相关函数，没有实现算子融合

```python
python ./tests/eval/eval_gsm8k.py --batch-size 4 --max-seq-len 1024 --prefill-step 256 --max-new-tokens 1024 --out-dir "./tests/eval/output"
```
### Result

| Accuracy              | Invalid              | Latency   | Output throughput（est.） |
| --------------------- | -------------------- | --------- | ----------------------- |
| 0.260（26/100 correct） | 0.000（0/100 invalid） | 272.191 s | 45.3 token/s            |
