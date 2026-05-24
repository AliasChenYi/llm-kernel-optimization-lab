# LLM Kernel Optimization Lab

这是一个面向 NVIDIA V100 的 LLM GPU kernel 开发与性能测试实验库。

第一阶段聚焦两个典型的 reduction 类 kernel：

- RMSNorm
- row-wise Softmax

每个 operator 都包含：

- PyTorch reference implementation
- Triton implementation
- correctness tests
- microbenchmarks

项目中已经包含 CUDA C++ extension 脚手架，可用于后续补充 CUDA 版本 kernel。

## 目标硬件

- GPU: NVIDIA Tesla V100 32GB
- Compute capability: sm_70
- 第一阶段优先 dtype: fp32 和 fp16

V100 不支持 bf16 tensor cores，因此 bf16 暂不作为第一阶段目标。

## 快速开始

在服务器上创建 Python 环境，并以 editable 模式安装当前项目：

```bash
cd /home/chendepeng/workspace/llm-kernel-optimization-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install triton pytest tabulate
python -m pip install -e .
```

验证环境：

```bash
source scripts/env.sh
python scripts/check_env.py
pytest -q
```

运行 benchmark：

```bash
source scripts/env.sh
python benchmarks/benchmark_rmsnorm.py
python benchmarks/benchmark_softmax.py
```

在共享 V100 服务器上运行 benchmark 前，建议优先选择负载最低的 GPU：

```bash
CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_rmsnorm.py --dtype fp16
```

构建并 smoke-test CUDA extension：

```bash
cd extensions/cuda
source ../../scripts/env.sh
../../.venv/bin/python setup.py build_ext --inplace
cd ../..
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/smoke_cuda_extension.py
```

PyTorch 最新安装命令可参考官方 selector：
https://pytorch.org/get-started/locally/

Triton 安装说明可参考官方文档：
https://triton-lang.org/main/getting-started/installation.html

## 项目结构

```text
benchmarks/                 Microbenchmarks
docs/                       环境配置、profiling 和 benchmark 文档
extensions/cuda/            CUDA C++ extension 脚手架
scripts/                    环境检查与 smoke test 脚本
src/llm_kernel_lab/         Python package
tests/                      Correctness tests
```

## 阶段规划

1. 使用 Triton 实现 RMSNorm 和 Softmax，并补齐 correctness tests 与 benchmark reports。
2. 在 `nvcc` 可用后补充 RMSNorm 和 Softmax 的 CUDA C++ 版本。
3. 实现 RoPE 和 SwiGLU fused kernels。
4. 实现 int8/int4 LLM inference 路径中的 dequantization kernels。
5. 使用 Nsight Compute 等工具沉淀 profiling notes，记录关键 tuning decisions。

## Profiling

RMSNorm profiling 的工具选择、指标含义和命令入口见：

```text
docs/rmsnorm_profiling_guide.md
```

## 当前服务器环境

- Python virtualenv: `.venv`
- PyTorch: `2.5.1+cu121`
- Triton: `3.1.0`
- CUDA Toolkit: `/usr/local/cuda-12.2`
- user-local Python headers: `/home/chendepeng/.local/python3.10-dev-root`
