from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on machines without Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, hidden_size: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden_size

        x = tl.load(x_ptr + row_id * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        variance = tl.sum(x * x, axis=0) / hidden_size
        y = x * tl.rsqrt(variance + eps) * w

        tl.store(y_ptr + row_id * hidden_size + offsets, y, mask=mask)


def _num_warps(block: int) -> int:
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


def rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Triton RMSNorm for contiguous tensors shaped [..., hidden_size]."""
    if triton is None:
        raise RuntimeError("Triton is not installed.")
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors.")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight length must match x.shape[-1].")

    x_2d = x.contiguous().view(-1, x.shape[-1])
    weight = weight.contiguous()
    y = torch.empty_like(x_2d)

    hidden_size = x_2d.shape[-1]
    block = triton.next_power_of_2(hidden_size)
    if block > 65536:
        raise ValueError(f"hidden_size={hidden_size} is too large for the single-block RMSNorm kernel.")

    grid = (x_2d.shape[0],)
    _rmsnorm_kernel[grid](
        x_2d,
        weight,
        y,
        hidden_size,
        eps,
        block,
        num_warps=_num_warps(block),
    )
    return y.view_as(x)

