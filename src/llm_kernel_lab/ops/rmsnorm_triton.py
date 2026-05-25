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
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
        cache_weight: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden_size

        x = tl.load(x_ptr + row_id * hidden_size + offsets, mask=mask, other=0.0).to(tl.float32)
        if cache_weight:
            w = tl.load(w_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        else:
            w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        variance = tl.sum(x * x, axis=0) / hidden_size
        y = x * tl.rsqrt(variance + eps) * w

        tl.store(y_ptr + row_id * hidden_size + offsets, y, mask=mask)

    @triton.jit
    def _rmsnorm_two_pass_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
        cache_weight: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden_size
        row_offsets = row_id * hidden_size + offsets

        x_for_var = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x_for_var * x_for_var, axis=0) / hidden_size
        inv_rms = tl.rsqrt(variance + eps)

        x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        if cache_weight:
            w = tl.load(w_ptr + offsets, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
        else:
            w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w

        tl.store(y_ptr + row_offsets, y, mask=mask)


def _num_warps(block: int, dtype: torch.dtype) -> int:
    if dtype == torch.float16:
        if block <= 2048:
            return 4
        if block <= 4096:
            return 8
        return 16

    if block <= 1024:
        return 4
    if block <= 2048:
        return 8
    if block <= 4096:
        return 16
    return 4


def rms_norm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    variant: str = "default",
    num_warps: int | None = None,
) -> torch.Tensor:
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

    if variant not in {"default", "cache_weight", "two_pass", "two_pass_cache_weight"}:
        raise ValueError(f"unsupported RMSNorm variant: {variant}")

    kernel = _rmsnorm_two_pass_kernel if variant.startswith("two_pass") else _rmsnorm_kernel
    cache_weight = variant.endswith("cache_weight")
    grid = (x_2d.shape[0],)
    kernel[grid](
        x_2d,
        weight,
        y,
        hidden_size,
        eps,
        block,
        cache_weight,
        num_warps=num_warps or _num_warps(block, x_2d.dtype),
    )
    return y.view_as(x)
