from __future__ import annotations

import argparse

import torch
import triton
from tabulate import tabulate

from llm_kernel_lab.ops import softmax_ref, softmax_triton


def _gbps(num_bytes: int, ms: float) -> float:
    return num_bytes / (ms * 1e-3) / 1e9


def _parse_warps(values: list[int]) -> list[int | None]:
    return [None if value == 0 else value for value in values]


def benchmark(
    batch: int,
    hidden: int,
    dtype: torch.dtype,
    variants: list[str],
    num_warps_values: list[int | None],
) -> list[dict[str, float | str | int]]:
    x = torch.randn((batch, hidden), device="cuda", dtype=dtype)

    results = []
    candidates: list[tuple[str, int | None, object]] = [("torch", None, lambda: softmax_ref(x))]
    for variant in variants:
        for num_warps in num_warps_values:
            label = f"triton:{variant}"
            if num_warps is not None:
                label = f"{label}:w{num_warps}"
            candidates.append((label, num_warps, lambda variant=variant, num_warps=num_warps: softmax_triton(
                x,
                variant=variant,
                num_warps=num_warps,
            )))

    for name, num_warps, fn in candidates:
        y = fn()
        torch.cuda.synchronize()
        ms = triton.testing.do_bench(fn)
        bytes_moved = x.numel() * x.element_size() * 2
        results.append(
            {
                "op": name,
                "batch": batch,
                "hidden": hidden,
                "dtype": str(dtype).replace("torch.", ""),
                "warps": "auto" if num_warps is None else num_warps,
                "ms": round(ms, 4),
                "GB/s": round(_gbps(bytes_moved, ms), 2),
                "checksum": float(y.float().sum().item()),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--variants", nargs="+", default=["default"], choices=["default", "exp2"])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[0], help="Use 0 for the default heuristic.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarks.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    num_warps_values = _parse_warps(args.num_warps)
    rows = []
    for hidden in args.hidden:
        rows.extend(benchmark(args.batch, hidden, dtype, args.variants, num_warps_values))

    print(tabulate(rows, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()
