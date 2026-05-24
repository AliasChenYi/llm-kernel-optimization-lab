from __future__ import annotations

import argparse

import torch
import triton
from tabulate import tabulate

from llm_kernel_lab.ops import softmax_ref, softmax_triton


def _gbps(num_bytes: int, ms: float) -> float:
    return num_bytes / (ms * 1e-3) / 1e9


def benchmark(batch: int, hidden: int, dtype: torch.dtype) -> list[dict[str, float | str | int]]:
    x = torch.randn((batch, hidden), device="cuda", dtype=dtype)

    results = []
    for name, fn in [
        ("torch", lambda: softmax_ref(x)),
        ("triton", lambda: softmax_triton(x)),
    ]:
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
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarks.")

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    rows = []
    for hidden in args.hidden:
        rows.extend(benchmark(args.batch, hidden, dtype))

    print(tabulate(rows, headers="keys", tablefmt="github"))


if __name__ == "__main__":
    main()

