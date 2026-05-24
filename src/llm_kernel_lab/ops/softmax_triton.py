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
    def _softmax_kernel(x_ptr, y_ptr, hidden_size: tl.constexpr, block: tl.constexpr):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden_size

        x = tl.load(x_ptr + row_id * hidden_size + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        x = x - tl.max(x, axis=0)
        numerator = tl.exp(x)
        denominator = tl.sum(numerator, axis=0)
        y = numerator / denominator

        tl.store(y_ptr + row_id * hidden_size + offsets, y, mask=mask)


def _num_warps(block: int) -> int:
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton row-wise softmax for contiguous tensors shaped [..., hidden_size]."""
    if triton is None:
        raise RuntimeError("Triton is not installed.")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor.")

    x_2d = x.contiguous().view(-1, x.shape[-1])
    y = torch.empty_like(x_2d)

    hidden_size = x_2d.shape[-1]
    block = triton.next_power_of_2(hidden_size)
    if block > 65536:
        raise ValueError(f"hidden_size={hidden_size} is too large for the single-block Softmax kernel.")

    grid = (x_2d.shape[0],)
    _softmax_kernel[grid](x_2d, y, hidden_size, block, num_warps=_num_warps(block))
    return y.view_as(x)

