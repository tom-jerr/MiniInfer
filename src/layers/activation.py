from collections.abc import Callable

import torch


def silu(x: torch.Tensor) -> torch.Tensor:
  """SiLU activation function, also known as the swish function.
  silu(x) = x * sigmoid(x)
  平滑且连续可导，非单调性

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying the SiLU activation function.
  """
  return x * torch.sigmoid(x)


def relu(x: torch.Tensor) -> torch.Tensor:
  """ReLU activation function.
  relu(x) = max(0, x)

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying the ReLU activation function.
  """
  return torch.relu(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
  """GELU activation function.
  Gaussian Error Linear Unit: gelu(x) = x * Φ(x)
  where Φ(x) is the cumulative distribution function of the standard normal distribution

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying the GELU activation function.
  """
  return torch.nn.functional.gelu(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
  """Tanh activation function.
  tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying the Tanh activation function.
  """
  return torch.tanh(x)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
  """Sigmoid activation function.
  sigmoid(x) = 1 / (1 + exp(-x))

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying the Sigmoid activation function.
  """
  return torch.sigmoid(x)


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
  """Leaky ReLU activation function.
  leaky_relu(x) = max(0, x) + negative_slope * min(0, x)

  Args:
      x (torch.Tensor): Input tensor.
      negative_slope (float): Controls the angle of the negative slope. Default is 0.01.

  Returns:
      torch.Tensor: Output tensor after applying the Leaky ReLU activation function.
  """
  return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)


def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
  """Swish activation function (generalized form of SiLU).
  swish(x) = x * sigmoid(beta * x)

  Args:
      x (torch.Tensor): Input tensor.
      beta (float): Scaling parameter. When beta=1, swish is equivalent to SiLU. Default is 1.0.

  Returns:
      torch.Tensor: Output tensor after applying the Swish activation function.
  """
  return x * torch.sigmoid(beta * x)


def mish(x: torch.Tensor) -> torch.Tensor:
  """Mish activation function.
  mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

  Args:
      x (torch.Tensor): Input tensor.

  Returns:
      torch.Tensor: Output tensor after applying the Mish activation function.
  """
  return x * torch.tanh(torch.nn.functional.softplus(x))


def glu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
  """Gated Linear Unit activation function.
  glu(x) = x[:, :n] * sigmoid(x[:, n:])
  Splits the input in half along the specified dimension.

  Args:
      x (torch.Tensor): Input tensor.
      dim (int): Dimension along which to split the input. Default is -1.

  Returns:
      torch.Tensor: Output tensor after applying the GLU activation function.
  """
  return torch.nn.functional.glu(x, dim=dim)


# ============================================================================
# Activation Implementation Registry
# ============================================================================

# 静态映射：字符串 -> activation实现函数
ACTIVATION_IMPLEMENTATIONS: dict[str, Callable] = {
  "silu": silu,
  "swish": silu,  # SiLU is also known as Swish
  "relu": relu,
  "gelu": gelu,
  "tanh": tanh,
  "sigmoid": sigmoid,
  "leaky_relu": leaky_relu,
  "mish": mish,
  "glu": glu,
}


def get_activation(name: str) -> Callable:
  """
  根据名称获取activation实现函数

  Args:
      name: activation实现的名称，支持以下选项：
          - "silu" 或 "swish": SiLU/Swish激活函数
          - "relu": ReLU激活函数
          - "gelu": GELU激活函数
          - "tanh": Tanh激活函数
          - "sigmoid": Sigmoid激活函数
          - "leaky_relu": Leaky ReLU激活函数
          - "mish": Mish激活函数
          - "glu": Gated Linear Unit激活函数

  Returns:
      对应的activation实现函数

  Raises:
      ValueError: 如果提供的名称不在支持的实现中

  Examples:
      >>> act_fn = get_activation("silu")
      >>> output = act_fn(input_tensor)
      >>> # 直接使用
      >>> output = get_activation("gelu")(input_tensor)
  """
  if name not in ACTIVATION_IMPLEMENTATIONS:
    available = ", ".join(ACTIVATION_IMPLEMENTATIONS.keys())
    raise ValueError(
      f"Unknown activation implementation: '{name}'. " f"Available options: {available}"
    )
  return ACTIVATION_IMPLEMENTATIONS[name]


def apply_activation(x: torch.Tensor, activation: str = "silu", **kwargs) -> torch.Tensor:
  """
  统一的激活函数接口，支持通过字符串选择不同的实现

  Args:
      x: Input tensor
      activation: 激活函数名称，可选 "silu", "relu", "gelu", "tanh", "sigmoid",
                 "leaky_relu", "mish", "glu"
      **kwargs: 传递给激活函数的额外参数（例如 leaky_relu 的 negative_slope）

  Returns:
      Output tensor after applying the activation function

  Examples:
      >>> # 使用SiLU激活
      >>> output = apply_activation(x, activation="silu")
      >>> # 使用Leaky ReLU并指定negative_slope
      >>> output = apply_activation(x, activation="leaky_relu", negative_slope=0.1)
      >>> # 使用GELU激活
      >>> output = apply_activation(x, activation="gelu")
  """
  act_fn = get_activation(activation)
  # 检查函数是否接受额外参数
  if kwargs:
    return act_fn(x, **kwargs)
  return act_fn(x)
