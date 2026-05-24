# RMSNorm Profiling Guide

本文档说明如何在当前 V100 服务器上查看 RMSNorm 优化相关指标。

## 使用哪些工具

推荐三个层次的 profiling 工具：

- `torch.profiler`: 看 PyTorch operator 级别的耗时、kernel launch 数量、CUDA time 和 memory 行为。
- `Nsight Systems` / `nsys`: 看整体 timeline，包括 CPU 调度、CUDA API、kernel launch 间隔、NVTX range。
- `Nsight Compute` / `ncu`: 看单个 CUDA kernel 的硬件指标，例如 DRAM throughput、L2 throughput、achieved occupancy、register usage、shared memory usage、warp execution efficiency。

官方文档参考：

- Nsight Compute CLI: https://docs.nvidia.com/nsight-compute/NsightComputeCli/
- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/
- PyTorch Profiler: https://docs.pytorch.org/docs/stable/profiler.html

## 环境准备

进入项目目录：

```bash
cd /home/chendepeng/workspace/llm-kernel-optimization-lab
source scripts/env.sh
```

服务器的 CUDA Toolkit 已包含：

```text
/usr/local/cuda-12.2/bin/ncu
/usr/local/cuda-12.2/bin/nsys
```

当前验证状态：

- `torch.profiler`: 可用。
- `Nsight Systems` / `nsys`: 可用，已成功生成 `.nsys-rep` 和 `.sqlite`。
- `Nsight Compute` / `ncu`: 工具已安装，但当前用户没有 GPU performance counters 权限，运行时会触发 `ERR_NVGPUCTRPERM`。

如果服务器上 GPU 0 或 GPU 1 正忙，建议使用 GPU 2：

```bash
export CUDA_VISIBLE_DEVICES=2
```

## 快速查看 PyTorch 层耗时

Triton RMSNorm:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/profile_rmsnorm_torch.py --impl triton --dtype fp16 --hidden 4096
```

PyTorch reference:

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/profile_rmsnorm_torch.py --impl torch --dtype fp16 --hidden 4096
```

输出表中重点看：

- `cuda_time_total`: CUDA 总耗时。
- `cuda_time_avg`: 单次平均 CUDA 耗时。
- `# of Calls`: kernel 或 operator 调用次数。
- `Self CUDA Mem`: 当前 operator 自身显存分配情况。

PyTorch reference 通常会出现多个 aten operator，例如 `aten::pow`、`aten::mean`、`aten::mul`。Triton 版本通常集中到一个自定义 kernel，因此 launch 数量和中间显存读写更少。

脚本会导出 Chrome trace：

```text
results/profiling/torch_rmsnorm_trace.json
```

可以在 Chrome 浏览器打开：

```text
chrome://tracing
```

## 使用 Nsight Systems 看 timeline

默认 profiling Triton fp16 RMSNorm:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/profile_rmsnorm_nsys.sh
```

修改参数：

```bash
CUDA_VISIBLE_DEVICES=2 IMPL=torch DTYPE=fp16 HIDDEN=4096 ITERS=20 bash scripts/profile_rmsnorm_nsys.sh
CUDA_VISIBLE_DEVICES=2 IMPL=triton DTYPE=fp32 HIDDEN=8192 ITERS=20 bash scripts/profile_rmsnorm_nsys.sh
```

输出文件位于：

```text
results/profiling/*.nsys-rep
```

重点看：

- CPU 到 GPU 的 kernel launch 间隔是否明显。
- 一个 RMSNorm iteration 里有多少 CUDA kernel。
- NVTX range `rmsnorm_triton_fp16_h4096` 内部 kernel 是否密集。
- PyTorch reference 是否出现多个小 kernel 串行执行。

`nsys` 适合回答“整体时间花在哪里”，不适合深入回答“单个 kernel 为什么慢”。

## 使用 Nsight Compute 看 kernel 硬件指标

默认 profiling Triton fp16 RMSNorm:

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/profile_rmsnorm_ncu.sh
```

修改参数：

```bash
CUDA_VISIBLE_DEVICES=2 IMPL=triton DTYPE=fp16 HIDDEN=8192 ITERS=10 bash scripts/profile_rmsnorm_ncu.sh
CUDA_VISIBLE_DEVICES=2 IMPL=triton DTYPE=fp32 HIDDEN=4096 ITERS=10 bash scripts/profile_rmsnorm_ncu.sh
```

输出文件位于：

```text
results/profiling/*.ncu-rep
```

可以用 CLI 查看摘要：

```bash
ncu --import results/profiling/ncu_rmsnorm_triton_fp16_b4096_h4096.ncu-rep
```

也可以把 `.ncu-rep` 下载到本地，用 `ncu-ui` 打开。

重点看以下 sections：

- `SpeedOfLight`: 判断 SM throughput 和 memory throughput 离硬件峰值有多远。
- `MemoryWorkloadAnalysis`: 看 DRAM/L2 访问、cache hit、memory pipe 压力。
- `Occupancy`: 看 achieved occupancy、theoretical occupancy、register usage 和 block/warp 配置。

RMSNorm 通常是 memory-bandwidth/reduction bound，而不是 FLOPS bound。因此优先关注 DRAM throughput、L2 hit rate、occupancy 和 register pressure。

## 常见问题

### ncu 报权限错误

如果看到类似 `ERR_NVGPUCTRPERM`，说明当前用户没有访问 GPU performance counters 的权限。需要管理员放开 NVIDIA profiling 权限，或使用管理员允许的 profiling 节点。

当前服务器已复现该问题：

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

这不是代码问题，也不是 Triton 问题，而是 NVIDIA driver 对 performance counters 的访问限制。NVIDIA 官方说明见：

```text
https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

Linux 上通常需要管理员配置 NVIDIA kernel module 参数，例如允许非管理员访问 performance counters。具体方式应由服务器管理员根据集群策略处理。

在权限放开前，可以先使用：

- `torch.profiler` 分析 operator 级别时间和 kernel launch 数量。
- `nsys` 分析 timeline、CUDA API、NVTX range 和 kernel launch gap。
- benchmark 脚本观察端到端性能变化。

### ncu 很慢

`ncu` 会 replay kernel 来采集硬件计数器，比普通 benchmark 慢很多。建议先用较小 `ITERS`，例如：

```bash
CUDA_VISIBLE_DEVICES=2 ITERS=5 bash scripts/profile_rmsnorm_ncu.sh
```

### nsys 生成文件太大

`nsys` 适合短时间采样。不要对长 benchmark 直接全程 profiling，建议用默认脚本里的 `cudaProfilerStart/Stop` capture range。

## RMSNorm 优化时的观察顺序

1. 先用 benchmark 确认端到端耗时是否真的变化。
2. 用 `torch.profiler` 确认 kernel launch 数量和 PyTorch operator 拆分情况。
3. 用 `nsys` 看 timeline，确认没有 CPU launch gap 或额外同步。
4. 用 `ncu` 看单个 Triton kernel 的 DRAM throughput、L2 behavior、achieved occupancy 和 register usage。
5. 调整 `num_warps`、`BLOCK_SIZE`、cache hint 或 CUDA reduction 结构，再回到 benchmark 验证。
