# Optimization Log: 2026-05-25

目标：围绕 RMSNorm 和 Softmax 在 V100 上做第一轮 Triton kernel tuning，并记录尝试、结果和结论。

## 实验环境

- Host: `dinglab-v100`
- GPU: Tesla V100-DGXS-32GB, sm_70
- PyTorch: `2.5.1+cu121`
- Triton: `3.1.0`
- CUDA Toolkit: `/usr/local/cuda-12.2`

运行前统一执行：

```bash
cd /home/chendepeng/workspace/llm-kernel-optimization-lab
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=2
```

## 优化点

### 1. RMSNorm `num_warps` sweep

原始实现使用静态 heuristic：

- block <= 1024: 4 warps
- block <= 4096: 8 warps
- block > 4096: 16 warps

本轮增加 `num_warps` 参数，用于在 benchmark 中显式 sweep。目标是观察 V100 上不同 hidden size 是否需要不同 warp 配置。

### 2. RMSNorm `cache_weight`

RMSNorm 的 `weight` 对所有 row 共享。新增 `cache_weight` 变体，对 `weight` load 使用 Triton cache hint：

```python
tl.load(..., cache_modifier=".ca")
```

目标是提高 `weight` 在 cache 中的复用概率，尤其观察较大 batch 下是否有收益。

### 3. RMSNorm `two_pass`

原始 kernel 一次 load `x` 后保留整行向量，完成 variance reduction 和 output store。该方式 global memory traffic 较少，但可能带来较高 register pressure。

新增 `two_pass` 变体：

1. 第一次 load `x` 计算 variance。
2. 第二次 load `x` 计算 output。

目标是用更多 global load 换更低 register pressure，观察 hidden size 较大时是否更快。

### 4. Softmax `num_warps` sweep

Softmax 同样增加 `num_warps` 参数，用于观察 512/1024/2048/4096 hidden size 下的 warp 配置敏感性。

### 5. Softmax `exp2`

新增 `exp2` 变体：

```python
x = (x - max(x)) * log2(e)
numerator = exp2(x)
```

目标是测试 `tl.exp2` 在 V100 上是否比 `tl.exp` 更快，同时保持 softmax 数学等价。

## 运行命令

RMSNorm sweep:

```bash
CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_rmsnorm.py \
  --dtype fp16 \
  --variants default cache_weight two_pass two_pass_cache_weight \
  --num-warps 4 8 16
```

Softmax sweep:

```bash
CUDA_VISIBLE_DEVICES=2 python benchmarks/benchmark_softmax.py \
  --dtype fp16 \
  --variants default exp2 \
  --num-warps 4 8 16
```

## 结果

### Correctness

所有新增变体均通过 correctness tests：

```text
36 passed in 9.04s
```

### RMSNorm fp16 最优结果

| hidden | torch ms | best Triton variant | best warps | best ms | best GB/s | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 1024 | 0.2637 | cache_weight | 4 | 0.0258 | 651.10 | `cache_weight` 有轻微收益，4 warps 最优 |
| 2048 | 0.4728 | default | 4 | 0.0459 | 730.85 | 原 heuristic 的 8 warps 偏多，4 warps 更好 |
| 4096 | 0.9057 | cache_weight | 8 | 0.0874 | 767.83 | 8 warps 最优，`cache_weight` 收益可忽略 |
| 8192 | 1.7640 | default | 16 | 0.1692 | 793.22 | 16 warps 略优 |

### RMSNorm fp32 最优结果

| hidden | torch ms | best Triton variant | best warps | best ms | best GB/s | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 1024 | 0.1669 | default | 4 | 0.0461 | 728.59 | 4 warps 最优 |
| 2048 | 0.3144 | default/cache_weight | 8 | 0.0868 | 773.31 | 8 warps 最优 |
| 4096 | 0.5980 | default | 16 | 0.1682 | 797.84 | 16 warps 最优 |
| 8192 | 1.1700 | default/cache_weight | 4 | 0.3347 | 802.19 | 4 warps 略优，差距很小 |

### Softmax fp16 最优结果

| hidden | torch ms | best Triton variant | best warps | best ms | best GB/s | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 512 | 0.0689 | default | 4 | 0.0157 | 532.97 | 4 warps 明显最优 |
| 1024 | 0.1309 | exp2/default | 4 | 0.0255 | 657.28 | `exp2` 与 default 基本持平 |
| 2048 | 0.2474 | default | 4 | 0.0448 | 748.98 | 原 heuristic 的 8 warps 偏多 |
| 4096 | 0.4693 | default | 8 | 0.0855 | 784.95 | 8 warps 最优 |

### Softmax fp32 最优结果

| hidden | torch ms | best Triton variant | best warps | best ms | best GB/s | 结论 |
|---:|---:|---|---:|---:|---:|---|
| 512 | 0.0268 | exp2 | 4 | 0.0254 | 661.10 | `exp2` 有轻微收益 |
| 1024 | 0.0510 | default | 4 | 0.0448 | 749.17 | 4 warps 最优 |
| 2048 | 0.0962 | default/exp2 | 8 | 0.0855 | 785.34 | 8 warps 最优 |
| 4096 | 0.1745 | default/exp2 | 16 | 0.1673 | 802.44 | 16 warps 最优 |

## 结论

1. `num_warps` 是本轮最有效的优化点。RMSNorm 和 Softmax 都表现出明显的 hidden size 与 dtype 相关性，因此已将默认 heuristic 改为 dtype-aware。
2. RMSNorm `cache_weight` 只在 fp16 hidden=1024 上有约 1.5% 的轻微收益，其余场景基本持平。原因可能是 `weight` 本身已能较好利用 L2 cache，显式 cache hint 对该 workload 影响有限。
3. RMSNorm `two_pass` 没有稳定收益。额外读取一次 `x` 增加了 memory traffic，而当前 kernel 已经接近 memory bandwidth bound，因此降低 register pressure 的潜在收益没有抵消额外 load 成本。
4. Softmax `exp2` 与默认 `tl.exp` 基本持平，只在少数 fp32 小 hidden 上有轻微收益，不建议作为默认实现替换。
5. 后续 RMSNorm 深入优化建议转向 CUDA C++ 版本的 warp-level reduction、vectorized load/store，以及在 `ncu` performance counters 权限放开后观察 register usage、achieved occupancy 和 memory throughput。

## 已采用的默认 heuristic

RMSNorm:

- fp16: hidden <= 2048 使用 4 warps，hidden <= 4096 使用 8 warps，否则使用 16 warps。
- fp32: hidden <= 1024 使用 4 warps，hidden <= 2048 使用 8 warps，hidden <= 4096 使用 16 warps，否则使用 4 warps。

Softmax:

- fp16: hidden <= 2048 使用 4 warps，hidden <= 4096 使用 8 warps，否则使用 16 warps。
- fp32: hidden <= 1024 使用 4 warps，hidden <= 2048 使用 8 warps，否则使用 16 warps。

## 默认路径复测

采用 dtype-aware heuristic 后，普通 benchmark 命令无需显式传入 `--num-warps` 即可使用新配置。

RMSNorm fp16:

| hidden | torch ms | triton default ms | triton GB/s |
|---:|---:|---:|---:|
| 1024 | 0.2643 | 0.0263 | 638.28 |
| 2048 | 0.4794 | 0.0461 | 727.99 |
| 4096 | 0.9064 | 0.0873 | 768.61 |
| 8192 | 1.7642 | 0.1692 | 793.17 |

RMSNorm fp32:

| hidden | torch ms | triton default ms | triton GB/s |
|---:|---:|---:|---:|
| 1024 | 0.1672 | 0.0462 | 726.85 |
| 2048 | 0.3143 | 0.0868 | 773.02 |
| 4096 | 0.5987 | 0.1682 | 797.89 |
| 8192 | 1.1700 | 0.3347 | 802.02 |

Softmax fp16:

| hidden | torch ms | triton default ms | triton GB/s |
|---:|---:|---:|---:|
| 512 | 0.0714 | 0.0164 | 512.00 |
| 1024 | 0.1343 | 0.0259 | 646.84 |
| 2048 | 0.2511 | 0.0448 | 749.51 |
| 4096 | 0.4693 | 0.0856 | 784.40 |

Softmax fp32:

| hidden | torch ms | triton default ms | triton GB/s |
|---:|---:|---:|---:|
| 512 | 0.0271 | 0.0259 | 647.55 |
| 1024 | 0.0510 | 0.0448 | 749.05 |
| 2048 | 0.0962 | 0.0856 | 784.32 |
| 4096 | 0.1746 | 0.1673 | 802.23 |
