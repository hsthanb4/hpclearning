# Nsight Compute 性能分析指南

本文档说明如何使用 Nsight Compute (ncu) 对 GEMM kernel 进行性能分析。

## 当前系统状态

**GPU**: NVIDIA GeForce RTX 3080 (GA102)
**Nsight Compute 版本**: 2025.1.1.0
**系统**: Linux x86_64

## 权限问题和解决方案

### 问题诊断

运行 ncu 时遇到权限错误:
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

**原因分析**:
```bash
# 检查 NVIDIA 驱动参数
$ cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
RmProfilingAdminOnly: 1
```

`RmProfilingAdminOnly: 1` 表示只有 root 用户可以访问 GPU 性能计数器。

---

## 解决方案

### 方案 1: 临时启用性能分析（推荐）

```bash
# 以 root 权限修改驱动参数
sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'

# 临时允许所有用户进行性能分析
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

**注意**: 这个设置在重启后会重置。

### 方案 2: 永久配置（需要重启）

创建或编辑 `/etc/modprobe.d/nvidia-profiling.conf`:

```bash
sudo tee /etc/modprobe.d/nvidia-profiling.conf <<EOF
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF

# 更新 initramfs
sudo update-initramfs -u

# 重启系统
sudo reboot
```

### 方案 3: 使用 sudo 运行（最简单）

```bash
# 切换到 build 目录
cd /home/lixiang/code/hpclearning/matmul/build

# 使用 sudo 运行 ncu
sudo /opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu \
  --set full \
  --launch-skip 0 \
  --launch-count 1 \
  -o gemm_profile_report \
  ./gemm_test 4
```

---

## 性能分析示例

### 示例 1: 快速性能概览

```bash
NCU=/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu

sudo $NCU \
  --print-summary per-kernel \
  --launch-skip 0 \
  --launch-count 1 \
  ./gemm_test 4
```

**输出信息**:
- Kernel 名称
- 执行时间
- 吞吐量
- 占用率

### 示例 2: 收集关键 GEMM 指标

```bash
sudo $NCU \
  --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
    smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
    smsp__sass_thread_inst_executed_op_dmul_pred_on.sum \
  --launch-skip 0 \
  --launch-count 1 \
  ./gemm_test 4
```

**关键指标解释**:
- `sm__throughput`: SM (Streaming Multiprocessor) 吞吐量百分比
- `dram__throughput`: DRAM 带宽利用率
- `l1tex__data_bank_conflicts`: Shared Memory Bank Conflicts
- `smsp__sass_thread_inst_executed`: 执行的指令数

### 示例 3: 生成完整分析报告（推荐）

```bash
sudo $NCU \
  --set full \
  --launch-skip 10 \
  --launch-count 1 \
  -o gemm_kernel4_full_profile \
  ./gemm_test 4
```

**生成文件**: `gemm_kernel4_full_profile.ncu-rep`

可以:
1. 下载到本地
2. 在 Nsight Compute GUI 中打开
3. 查看详细的性能分析、瓶颈识别、优化建议

---

## 常用 NCU 选项

### 基本选项

```bash
# 查看帮助
ncu --help

# 列出所有可用指标
ncu --query-metrics

# 列出所有可用的分析集
ncu --list-sets

# 查看特定指标的描述
ncu --query-metrics-description sm__throughput
```

### Kernel 过滤

```bash
# 只分析特定名称的 kernel
ncu --kernel-name "myKernel" ./program

# 使用正则表达式匹配
ncu --kernel-regex ".*gemm.*" ./program

# 跳过前 N 次启动，只分析后续的
ncu --launch-skip 10 --launch-count 1 ./program

# 分析所有 kernel 启动
ncu --launch-count-all ./program
```

### 输出控制

```bash
# 简洁输出
ncu --print-summary per-kernel ./program

# 详细输出
ncu --print-summary per-gpu ./program

# 输出到文件（可在 GUI 中打开）
ncu -o output_report ./program

# 导出为 CSV
ncu --csv ./program

# 导出为 JSON
ncu --json ./program
```

### 指标选择

```bash
# 使用预定义的指标集
ncu --set full ./program          # 完整指标
ncu --set detailed ./program      # 详细指标
ncu --set roofline ./program      # Roofline 分析

# 自定义指标
ncu --metrics metric1,metric2,metric3 ./program

# 指标加统计量
ncu --metrics "metric1.avg,metric2.max,metric3.sum" ./program
```

---

## 实用分析流程

### 第 1 步: 快速诊断

```bash
# 快速了解性能
sudo $NCU --print-summary per-kernel ./gemm_test 4
```

### 第 2 步: 识别瓶颈

```bash
# Roofline 分析
sudo $NCU --set roofline -o roofline_report ./gemm_test 4
```

在 GUI 中查看 Roofline Chart，判断是:
- **Memory-bound**: 受内存带宽限制
- **Compute-bound**: 受计算能力限制

### 第 3 步: 详细分析

```bash
# 完整分析
sudo $NCU --set full -o full_report ./gemm_test 4
```

在 GUI 中查看:
- **Speed of Light**: 各项资源利用率
- **Memory Workload Analysis**: 内存访问模式
- **Compute Workload Analysis**: 计算指令分布
- **Scheduler Statistics**: Warp 调度效率
- **Occupancy**: 占用率分析

### 第 4 步: 优化验证

修改代码后重新分析，对比优化前后的指标。

---

## RTX 3080 关键性能指标

### 硬件规格

- **架构**: Ampere (GA102)
- **SM 数量**: 68 个
- **CUDA Cores**: 8704 个
- **Tensor Cores**: 272 个 (第3代)
- **Base Clock**: 1440 MHz
- **Boost Clock**: 1710 MHz
- **Memory**: 10 GB GDDR6X
- **Memory Bandwidth**: 760 GB/s
- **TDP**: 320W

### 理论峰值性能

- **FP32**: 29.77 TFLOPS
- **FP16 (Tensor Core)**: 119 TFLOPS
- **INT8 (Tensor Core)**: 238 TOPS

### 关键优化目标

对于 GEMM kernel:

1. **Tensor Core 利用率**: 目标 > 80%
2. **Memory Bandwidth**: 目标 > 700 GB/s (> 90% 理论峰值)
3. **SM 利用率**: 目标 > 70%
4. **Occupancy**: 目标 > 50%
5. **Bank Conflicts**: 目标 = 0

---

## 常见问题排查

### Q1: ncu 报错 "command not found"

**解决**:
```bash
# 使用完整路径
/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu --version

# 或者添加到 PATH
export PATH=/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64:$PATH
```

### Q2: 报错 "ERR_NVGPUCTRPERM"

**原因**: 没有 GPU 性能计数器访问权限

**解决**: 参考本文档的"解决方案"部分

### Q3: GUI 无法打开 .ncu-rep 文件

**可能原因**:
- GUI 版本与 ncu 版本不匹配
- 文件损坏

**解决**:
```bash
# 检查文件完整性
file gemm_profile_report.ncu-rep

# 使用匹配版本的 GUI
# 下载地址: https://developer.nvidia.com/nsight-compute
```

### Q4: 分析时程序崩溃

**可能原因**:
- GPU 内存不足
- Kernel 有 bug
- 驱动问题

**解决**:
```bash
# 先在没有 profiler 的情况下测试
./gemm_test 4

# 检查 GPU 状态
nvidia-smi

# 查看系统日志
dmesg | tail -50
```

---

## 推荐学习资源

### 官方文档
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [Nsight Compute CLI Reference](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [CUDA Profiling Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)

### 教程
- [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/)
- [CUDA Training Series](https://www.nvidia.com/en-us/on-demand/cuda-training-series/)

### 视频
- [GTC Talks on Performance Analysis](https://www.nvidia.com/en-us/gtc/)
- [Nsight Compute Tutorial Videos](https://www.youtube.com/nvidia)

---

## 示例分析报告解读

### Roofline Chart 解读

```
                Compute Bound (上方区域)
                       |
     Performance       |    ← Your Kernel
          ↑           |   ○
          |          /|
          |        /  |
          |      /    |
          |    /      |
          |  /        |
          |/__________|___________→ Arithmetic Intensity
           Memory Bound (左下区域)
```

- **左下区域**: Memory-bound，需要优化内存访问
- **右上区域**: Compute-bound，需要提高计算密度
- **对角线**: Roofline，硬件理论极限

### Speed of Light 解读

```
Metric                        Value    Peak
─────────────────────────────────────────────
Memory Throughput            650 GB/s  760 GB/s  (85%)
Compute Throughput           22 TFLOPS 30 TFLOPS (73%)
SM Active Cycles             89%       100%
Warp Efficiency              92%       100%
```

**优化建议**:
- Memory < 80%: 考虑内存 coalescing、shared memory 使用
- Compute < 80%: 考虑提高算术强度、使用 Tensor Core
- SM Active < 80%: 考虑增加并行度、减少同步
- Warp Efficiency < 90%: 考虑减少分支发散

---

## 更新日志

- **2026-02-02**: 创建文档，适配 RTX 3080 和 Nsight Compute 2025.1.1
- 包含权限问题解决方案
- 添加详细的分析示例和性能指标

---

**文档维护**:
- 创建时间: 2026-02-02
- 适用系统: Linux x86_64
- 适用 GPU: NVIDIA Ampere 架构及以上
- Nsight Compute 版本: 2025.1.1+
