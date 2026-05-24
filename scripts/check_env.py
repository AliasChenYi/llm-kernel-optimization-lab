from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


def _version(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return "not installed"
    module = __import__(module_name)
    return getattr(module, "__version__", "installed")


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {_version('torch')}")
    print(f"triton: {_version('triton')}")
    print(f"nvcc: {shutil.which('nvcc') or 'not found'}")

    if importlib.util.find_spec("torch") is None:
        return

    import torch

    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch.version.cuda: {torch.version.cuda}")
        for idx in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(idx)
            print(f"gpu[{idx}]: {torch.cuda.get_device_name(idx)}, sm_{capability[0]}{capability[1]}")

    if shutil.which("nvidia-smi"):
        subprocess.run(["nvidia-smi"], check=False)


if __name__ == "__main__":
    main()

