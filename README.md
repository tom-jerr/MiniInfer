# MiniInfer

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**一个从零开始构建的轻量级高性能 LLM 推理引擎**

</div>

---

## 📝 项目简介

MiniInfer 是一个学习性质的大语言模型推理引擎，从零实现了现代 LLM 推理系统的核心组件。本项目旨在帮助开发者深入理解 LLM 推理的底层机制，包括注意力机制、KV 缓存、批处理调度等关键技术。

## 🙏 致谢

- \*\*[mini-sglang](https://github.com/sgl-project/mini-sglang) - 提供了 KV Cache 相关的设计思路和部分代码实现
- **[tiny-llm](https://github.com/skyzh/tiny-llm)** - 最初的项目框架代码参考
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** - 重构后项目框架代码参考
- **[vLLM](https://github.com/vllm-project/vllm)** - 高性能 LLM 推理引擎，PagedAttention 和 Continuous Batching 的创新实现给了我们很大启发
