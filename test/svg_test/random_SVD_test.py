from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
):
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        random_matrix = tensor.uniform_(-bound, bound, generator=generator)
        U, S, Vh =torch.linalg.svd(random_matrix)
        return U, S, Vh, random_matrix

def SVD_test():
    generator = torch.Generator(device="cpu").manual_seed(0)
    U, S, Vh, rm  = _kaiming_init((128, 512), generator=generator)
    S = S.unsqueeze(-1)
    print(S)
    if rm.size(0) != rm.size(1):
        S_diag = torch.zeros((rm.size(0), rm.size(1)))
        S_diag[:, :rm.size(0)] = torch.diag_embed(S)
    else:
        S_diag = torch.diag(S)

    # reconstructed_tensor = S * U @ Vh
    # reconstructed_tensor = Vh @ (S * U)
    reconstructed_tensor = U @ S_diag @ Vh
    reconstruction_error = torch.norm(rm - reconstructed_tensor, p='fro').item()

    print(reconstruction_error)

    plt.figure(figsize=(8, 6))
    plt.plot(S.cpu().numpy(), 'o-', label='Singular Values')
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 输入张量和权重
    A = torch.randn(128, 86, 768)
    B = torch.randn(768, 3072)

    # 执行线性变换
    result = F.linear(A, B.T)

    print(result.shape)  # 输出: torch.Size([128, 86, 3072])