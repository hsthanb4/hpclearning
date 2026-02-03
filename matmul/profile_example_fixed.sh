#!/bin/bash
# Nsight Compute 性能分析脚本 (修正版)
#
# 使用前请阅读: NCU_PROFILING_GUIDE.md
#
# 权限要求: 需要 root 权限访问 GPU 性能计数器
# 解决方案: sudo ./profile_example_fixed.sh

set -e  # 遇到错误立即退出

# 检测操作系统和架构
if [[ "$(uname -m)" == "x86_64" ]]; then
    NCU_PATH="/opt/nvidia/nsight-compute/2025.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu"
else
    echo "错误: 不支持的架构 $(uname -m)"
    exit 1
fi

# 检查 ncu 是否存在
if [ ! -f "$NCU_PATH" ]; then
    echo "错误: 找不到 Nsight Compute"
    echo "预期路径: $NCU_PATH"
    echo ""
    echo "请安装 NVIDIA Nsight Compute:"
    echo "https://developer.nvidia.com/nsight-compute"
    exit 1
fi

# 检查权限
echo "=== 检查 GPU 性能计数器权限 ==="
if [ -f "/proc/driver/nvidia/params" ]; then
    PROFILING_ADMIN=$(grep RmProfilingAdminOnly /proc/driver/nvidia/params | cut -d' ' -f2)
    echo "RmProfilingAdminOnly: $PROFILING_ADMIN"

    if [ "$PROFILING_ADMIN" == "1" ]; then
        if [ "$EUID" -ne 0 ]; then
            echo ""
            echo "警告: 需要 root 权限进行性能分析"
            echo "请使用: sudo $0"
            echo ""
            echo "或者参考 NCU_PROFILING_GUIDE.md 配置永久权限"
            exit 1
        fi
    fi
fi

# 设置工作目录
BUILD_DIR="/home/lixiang/code/hpclearning/matmul/build"
cd "$BUILD_DIR" || exit 1

echo "工作目录: $(pwd)"
echo "Nsight Compute: $NCU_PATH"
echo ""

# 检查可执行文件
if [ ! -f "./gemm_test" ]; then
    echo "错误: 找不到 gemm_test"
    echo "请先编译: cd .. && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

echo "=== 示例 1: 快速性能概览 ==="
echo "只分析第一个矩阵大小的第一次 kernel 调用"
echo ""
$NCU_PATH \
  --print-summary per-kernel \
  --launch-skip 0 \
  --launch-count 1 \
  ./gemm_test 4

echo ""
echo "=== 示例 2: 收集关键 GEMM 指标 ==="
echo "分析内存带宽、SM 利用率、Bank Conflicts 等"
echo ""
$NCU_PATH \
  --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --launch-skip 0 \
  --launch-count 1 \
  ./gemm_test 4

echo ""
echo "=== 示例 3: 生成完整分析报告 ==="
echo "生成可在 GUI 中打开的完整报告"
echo ""
$NCU_PATH \
  --set full \
  --launch-skip 10 \
  --launch-count 1 \
  -o gemm_kernel4_full_profile \
  ./gemm_test 4

echo ""
echo "=== 分析完成! ==="
echo ""
echo "生成的文件:"
ls -lh gemm_kernel4_full_profile.ncu-rep 2>/dev/null || echo "  (未生成报告文件)"
echo ""
echo "使用方法:"
echo "  1. 下载 gemm_kernel4_full_profile.ncu-rep 到本地"
echo "  2. 在 Nsight Compute GUI 中打开"
echo "  3. 查看详细的性能分析和优化建议"
echo ""
echo "更多信息请参考: NCU_PROFILING_GUIDE.md"
