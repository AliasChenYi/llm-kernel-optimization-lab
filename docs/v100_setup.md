# V100 Setup Notes

已观察到的服务器状态：

- Host: `dinglab-v100`
- GPUs: 3 x Tesla V100-DGXS-32GB
- Compute capability: 7.0
- Driver: 535.161.07
- Python: 3.10.12
- 当前默认 PATH 中缺少：`conda`、`pip`、`nvcc`

## Python Environment

服务器初始状态有 Python，但没有 `pip` 或 `python3.10-dev`。当前可工作的方案使用 user-local `pip`、项目内 `.venv`，以及 user-local Python headers 来支持 Triton runtime helper compilation。

使用过的 bootstrap 命令：

```bash
cd /tmp
wget -O get-pip.py https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py --user
python3 -m pip install --user virtualenv
cd /home/chendepeng/workspace/llm-kernel-optimization-lab
python3 -m virtualenv .venv
```

安装 runtime dependencies：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install triton pytest tabulate
python -m pip install -e .
```

Python dev headers 已在无 sudo 的情况下下载并解压到：

```bash
/home/chendepeng/.local/python3.10-dev-root
```

运行 Triton 代码前需要执行：

```bash
source scripts/env.sh
```

PyTorch 最新推荐安装命令可参考：
https://pytorch.org/get-started/locally/

Triton 稳定安装路径可参考：
https://triton-lang.org/main/getting-started/installation.html

## CUDA C++ Extension

`extensions/cuda` 下的 CUDA extension 需要 `nvcc`。当前 shell 默认找不到 `nvcc`，但 CUDA 12.2 可用路径为：

```bash
/usr/local/cuda-12.2/bin/nvcc
```

构建 extension 前请执行 `source scripts/env.sh`。针对 V100，编译目标使用 `sm_70`。

## V100 注意事项

- 第一阶段优先使用 fp16 和 fp32。
- bf16 不作为第一阶段目标。
- CUDA C++ 编译目标使用 `sm_70`。
- V100 支持 fp16 GEMM 风格 kernel 的 tensor cores，但 RMSNorm 和 Softmax 更偏向 reduction/memory-bandwidth oriented kernels。

