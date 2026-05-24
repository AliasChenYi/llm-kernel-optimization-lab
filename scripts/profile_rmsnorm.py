from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from llm_kernel_lab.ops import rms_norm_ref, rms_norm_triton


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _implementation(name: str, x: torch.Tensor, weight: torch.Tensor) -> Callable[[], torch.Tensor]:
    if name == "torch":
        return lambda: rms_norm_ref(x, weight)
    if name == "triton":
        return lambda: rms_norm_triton(x, weight)
    raise ValueError(f"unsupported implementation: {name}")


def _maybe_start_cuda_profiler(enabled: bool) -> None:
    if enabled:
        torch.cuda.cudart().cudaProfilerStart()


def _maybe_stop_cuda_profiler(enabled: bool) -> None:
    if enabled:
        torch.cuda.cudart().cudaProfilerStop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a single RMSNorm implementation.")
    parser.add_argument("--impl", choices=["torch", "triton"], default="triton")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket the measured loop with cudaProfilerStart/Stop for ncu/nsys capture ranges.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    dtype = _dtype(args.dtype)
    torch.manual_seed(0)
    x = torch.randn((args.batch, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.hidden,), device="cuda", dtype=dtype)
    fn = _implementation(args.impl, x, weight)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    _maybe_start_cuda_profiler(args.cuda_profiler_api)
    torch.cuda.nvtx.range_push(f"rmsnorm_{args.impl}_{args.dtype}_h{args.hidden}")
    for _ in range(args.iters):
        y = fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    _maybe_stop_cuda_profiler(args.cuda_profiler_api)

    print(
        {
            "impl": args.impl,
            "batch": args.batch,
            "hidden": args.hidden,
            "dtype": args.dtype,
            "iters": args.iters,
            "checksum": float(y.float().sum().item()),
        }
    )


if __name__ == "__main__":
    main()

