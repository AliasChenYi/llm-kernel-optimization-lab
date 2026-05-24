from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from llm_kernel_lab.ops import rms_norm_ref, rms_norm_triton


def main() -> None:
    parser = argparse.ArgumentParser(description="Run torch.profiler for RMSNorm.")
    parser.add_argument("--impl", choices=["torch", "triton"], default="triton")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="results/profiling/torch_rmsnorm_trace.json")
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    x = torch.randn((args.batch, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.hidden,), device="cuda", dtype=dtype)
    fn = (lambda: rms_norm_ref(x, weight)) if args.impl == "torch" else (lambda: rms_norm_triton(x, weight))

    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.iters):
            fn()
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace(str(out_path))
    print(f"Chrome trace: {out_path}")


if __name__ == "__main__":
    main()

