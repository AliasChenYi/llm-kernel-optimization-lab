from __future__ import annotations

import torch


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch RMSNorm reference for inputs shaped [..., hidden_size]."""
    if x.shape[-1] != weight.numel():
        raise ValueError("weight length must match x.shape[-1]")

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + eps)
    return (y * weight).to(dtype=x.dtype)


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """PyTorch row-wise softmax reference for inputs shaped [..., hidden_size]."""
    return torch.softmax(x.float(), dim=-1).to(dtype=x.dtype)

