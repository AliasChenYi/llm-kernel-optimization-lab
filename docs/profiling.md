# Profiling Notes

先从 wall-clock microbenchmarks 开始，等 kernel correctness 稳定后，再进入 Nsight Compute 分析。

常用 benchmark 命令：

```bash
python benchmarks/benchmark_rmsnorm.py --dtype fp16
python benchmarks/benchmark_rmsnorm.py --dtype fp32
python benchmarks/benchmark_softmax.py --dtype fp16
python benchmarks/benchmark_softmax.py --dtype fp32
```

reduction 类 kernel 常看的 Nsight Compute metrics：

- achieved occupancy
- dram throughput
- l2 throughput
- register usage
- shared memory usage
- warp execution efficiency

第一阶段目标不是立刻超过所有 library implementation。更有价值的目标是形成清晰的性能叙事：改了什么、为什么有变化，以及当前 kernel 正在逼近哪个 hardware limit。

