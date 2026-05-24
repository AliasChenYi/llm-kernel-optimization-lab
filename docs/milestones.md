# Milestones

## Milestone 1: Baseline Triton Kernels

- RMSNorm reference 和 Triton implementation
- Softmax reference 和 Triton implementation
- fp32/fp16 correctness tests
- 常见 LLM hidden size 的 microbenchmarks

## Milestone 2: CUDA C++ Implementations

- 构建 `extensions/cuda`
- 实现 RMSNorm CUDA baseline
- 补充 Softmax CUDA baseline
- 对比 PyTorch、Triton 和 CUDA C++ 性能

## Milestone 3: LLM Fusion Kernels

- RoPE
- SwiGLU
- fused residual + RMSNorm

## Milestone 4: Quantization Kernels

- int8 dequant
- int4 unpack + dequant
- fused dequant + activation

## Milestone 5: Profiling Reports

- benchmark tables
- Nsight Compute screenshots 或导出的 metrics
- 记录 block size、memory traffic、occupancy 和 numerical error 的 tuning notes

