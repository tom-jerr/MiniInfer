# MiniInfer C++/CUDA Extension

这是 MiniInfer 的 C++/CUDA 扩展模块，使用 CMake 构建系统和 pybind11 进行 Python 绑定。

## 项目结构

```
csrc/
├── CMakeLists.txt          # 主 CMake 配置文件
├── build.sh                # 构建脚本
├── setup.py                # 备用 setuptools 构建 (pip install)
├── bindings.cpp            # Python 绑定定义
├── cmake/
│   └── FindTorch.cmake     # PyTorch 查找模块
├── miniinfer_ext/
│   └── __init__.py         # Python 包入口
├── kernels/
│   ├── vector_add.h        # 头文件
│   ├── vector_add.cu       # CUDA 实现
│   └── rope.cu             # RoPE 实现 (待完成)
└── test/
    └── ...                 # 测试文件
```

## 构建方式

### 方式一：CMake 构建 (推荐)

```bash
cd kernels/csrc
chmod +x build.sh
./build.sh

# 或者手动构建
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### 方式二：pip 安装

```bash
cd kernels/csrc
pip install -e .
```

## 使用方法

### Python 中使用

```python
import torch
from miniinfer_ext import vector_add

# 创建张量
a = torch.randn(1024, device='cuda')
b = torch.randn(1024, device='cuda')

# 调用 CUDA 算子
result = vector_add(a, b)
```

### 从其他模块导入

```python
# 在 MiniInfer 的其他模块中
import sys
sys.path.append('/path/to/MiniInfer/kernels/csrc')
from miniinfer_ext import vector_add
```

## 添加新的算子

1. 在 `kernels/` 目录下创建头文件 (`.h`) 和实现文件 (`.cu` 或 `.cpp`)
2. 在 `CMakeLists.txt` 中添加源文件
3. 在 `bindings.cpp` 中添加 Python 绑定
4. 在 `miniinfer_ext/__init__.py` 中导出函数

### 示例：添加新算子

```cpp
// kernels/my_op.h
#pragma once
#include <torch/torch.h>

void my_op_cuda(torch::Tensor input, torch::Tensor output);
```

```cpp
// kernels/my_op.cu
#include <torch/torch.h>
#include "my_op.h"

__global__ void my_op_kernel(...) { ... }

void my_op_cuda(torch::Tensor input, torch::Tensor output) {
    // 实现
}
```

```cpp
// bindings.cpp 中添加
m.def("my_op", &my_op, "My custom operation");
```

## 依赖

- CMake >= 3.18
- CUDA Toolkit
- PyTorch (with CUDA support)
- Python >= 3.8

## 故障排除

### 找不到 torch/extension.h

确保 PyTorch 已正确安装，并且 CMake 能找到它：

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

### CUDA 架构不匹配

在 CMakeLists.txt 中修改 `CUDA_ARCHITECTURES`：

```cmake
set_target_properties(miniinfer_kernels PROPERTIES
    CUDA_ARCHITECTURES "80"  # 根据你的 GPU 调整
)
```

常见 GPU 架构：
- RTX 20 系列: 75
- RTX 30 系列: 86
- RTX 40 系列: 89
- A100: 80
- H100: 90
