from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    extension_dir = repo_root / "extensions" / "cuda"
    sys.path.insert(0, str(extension_dir))

    import llm_kernel_lab_cuda

    x = torch.randn((4, 1024), device="cuda", dtype=torch.float16)
    weight = torch.randn((1024,), device="cuda", dtype=torch.float16)
    y = llm_kernel_lab_cuda.rmsnorm_forward(x.contiguous(), weight.contiguous(), 1e-6)
    torch.cuda.synchronize()

    print({"shape": tuple(y.shape), "dtype": str(y.dtype), "finite": bool(torch.isfinite(y).all().item())})


if __name__ == "__main__":
    main()

