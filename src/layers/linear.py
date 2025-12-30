import torch


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
  """Compute the softmax of a tensor along a specified axis.
  softmax(x_i) = exp(x_i) / sum(exp(x_i))

  Args:
      x (torch.Tensor): Input tensor.
      axis (int): Axis along which to compute the softmax.

  Returns:
      torch.Tensor: Softmax of the input tensor along the specified axis.
  """
  # TODO(LZY): 这里自己实现的softmax不够精度，先用torch的版本
  return torch.softmax(x, dim=axis)


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
  """Apply a linear transformation to the incoming data: y = xW^T + b

  Args:
      x (torch.Tensor): Input tensor of shape (N, *, in_features).
      weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
      bias (torch.Tensor, optional): Bias tensor of shape (out_features,). Default is None.

  Returns:
      torch.Tensor: Output tensor of shape (N, *, out_features).
  """
  y = torch.matmul(x, weight.t())
  if bias is not None:
    y += bias
  return y
