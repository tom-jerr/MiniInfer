"""
测试activation函数统一接口
"""

import sys

import pytest
import torch

sys.path.append("..")

from miniinfer.layers.activation import (
    ACTIVATION_IMPLEMENTATIONS,
    apply_activation,
    gelu,
    get_activation,
    relu,
    sigmoid,
    silu,
    tanh,
)


def test_get_activation_valid():
    """测试获取有效的激活函数"""
    for name in ACTIVATION_IMPLEMENTATIONS.keys():
        fn = get_activation(name)
        assert callable(fn), f"{name} should return a callable"


def test_get_activation_invalid():
    """测试获取无效的激活函数应抛出异常"""
    with pytest.raises(ValueError):
        get_activation("invalid_activation")


def test_apply_activation_basic():
    """测试基本的激活函数应用"""
    x = torch.randn(2, 3, 4)

    # 测试每个激活函数
    for name in ["silu", "relu", "gelu", "tanh", "sigmoid"]:
        output = apply_activation(x, activation=name)
        assert output.shape == x.shape, f"Shape mismatch for {name}"
        assert not torch.isnan(output).any(), f"NaN detected in {name} output"


def test_silu_equivalence():
    """测试SiLU和Swish的等价性"""
    x = torch.randn(3, 4)

    silu_output = apply_activation(x, activation="silu")
    swish_output = apply_activation(x, activation="swish")

    assert torch.allclose(
        silu_output, swish_output
    ), "SiLU and Swish should be equivalent"


def test_activation_correctness():
    """测试激活函数的正确性"""
    x = torch.tensor([[-1.0, 0.0, 1.0]])

    # ReLU: max(0, x)
    relu_output = apply_activation(x, activation="relu")
    expected_relu = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(relu_output, expected_relu), "ReLU output incorrect"

    # Tanh: [-1, 1]
    tanh_output = apply_activation(x, activation="tanh")
    assert torch.all(tanh_output >= -1.0) and torch.all(
        tanh_output <= 1.0
    ), "Tanh should be in [-1, 1]"

    # Sigmoid: [0, 1]
    sigmoid_output = apply_activation(x, activation="sigmoid")
    assert torch.all(sigmoid_output >= 0.0) and torch.all(
        sigmoid_output <= 1.0
    ), "Sigmoid should be in [0, 1]"


def test_leaky_relu_parameters():
    """测试Leaky ReLU的参数传递"""
    x = torch.tensor([[-1.0, 0.0, 1.0]])

    output1 = apply_activation(x, activation="leaky_relu", negative_slope=0.1)
    output2 = apply_activation(x, activation="leaky_relu", negative_slope=0.2)

    # 负值部分应该不同
    assert not torch.allclose(
        output1, output2
    ), "Different negative_slope should give different outputs"


def test_gradient_flow():
    """测试梯度流是否正常"""
    x = torch.randn(2, 3, requires_grad=True)

    for name in ["silu", "relu", "gelu", "tanh"]:
        output = apply_activation(x, activation=name)
        loss = output.sum()
        loss.backward()

        if name != "relu":  # ReLU在某些点梯度为0
            assert x.grad is not None, f"Gradient should flow through {name}"

        x.grad = None  # 清除梯度以便下一次测试


def test_individual_functions():
    """测试单独导出的激活函数"""
    x = torch.randn(2, 3)

    # 直接调用函数
    output_silu = silu(x)
    output_relu = relu(x)
    output_gelu = gelu(x)
    output_tanh = tanh(x)
    output_sigmoid = sigmoid(x)

    # 通过接口调用并比较
    assert torch.allclose(output_silu, apply_activation(x, "silu"))
    assert torch.allclose(output_relu, apply_activation(x, "relu"))
    assert torch.allclose(output_gelu, apply_activation(x, "gelu"))
    assert torch.allclose(output_tanh, apply_activation(x, "tanh"))
    assert torch.allclose(output_sigmoid, apply_activation(x, "sigmoid"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
