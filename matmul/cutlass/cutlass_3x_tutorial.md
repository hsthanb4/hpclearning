# CUTLASS 3.x 教程：GEMM Kernel 设计的正交、可复用和可组合抽象

## 概述

CUTLASS 3.x 是 NVIDIA 推出的用于高性能 GEMM（通用矩阵乘法）kernel 设计的框架。它通过层次化、可组合的系统，在多个层级提供正交抽象，最大化了实现可能性的覆盖范围。

本文档将详细介绍 CUTLASS 3.x 的核心概念、设计理念和使用方法。

---

## 1. 设计理念

### 1.1 为什么需要 CUTLASS 3.x？

传统的 CUDA GEMM 实现存在以下挑战：

- **复杂性高**：直接编写高性能 GEMM kernel 需要深入理解硬件特性
- **可复用性差**：针对不同硬件架构需要重写大量代码
- **优化空间大**：不同的 tile 大小、调度策略、流水线设计导致组合爆炸

CUTLASS 3.x 通过**正交抽象**解决这些问题：
- **正交性**：不同层级的抽象相互独立，可以自由组合
- **可复用性**：底层组件可以在不同场景下重复使用
- **可组合性**：通过组合简单抽象构建复杂功能

### 1.2 硬件层次映射

CUTLASS 3.x 的设计镜像了 GPU 硬件层次结构：

```
Device (设备)
  └─ Kernel (内核/Grid)
      └─ CTA/Block (线程块)
          └─ Warp (线程束)
              └─ Thread (线程)
                  └─ Instruction (指令)
```

---

## 2. 五层抽象架构

CUTLASS 3.x 采用五层抽象，从底层硬件指令到高层设备接口：

### 2.1 Atom Layer (原子层)

**定义**：架构特定的基本指令单元

**核心组件**：
- `cute::Mma_Atom<>`：矩阵乘加原子操作
- `cute::Copy_Atom<>`：数据拷贝原子操作

**特点**：
- 直接对应硬件指令（如 Tensor Core 的 `wmma` 或 `mma` 指令）
- 针对特定架构优化（Ampere、Hopper、Blackwell 等）

**示例概念**：
```cpp
// Hopper 架构的 TensorCore MMA 原子
using MmaAtom = cute::Mma_Atom<
    SM90_16x8x16_F32F16F16F32_TN  // 指令类型
>;

// 异步拷贝原子（使用 TMA - Tensor Memory Accelerator）
using CopyAtom = cute::Copy_Atom<
    SM90_TMA_LOAD  // TMA 加载指令
>;
```

---

### 2.2 Tiled MMA/Copy (平铺 MMA/拷贝层)

**定义**：空间上的微内核，将原子操作平铺到更大的数据块

**核心组件**：
- `cute::TiledMma<>`：平铺的矩阵乘法
- `cute::TiledCopy<>`：平铺的数据拷贝

**功能**：
- 灵活的 tiling 策略
- 指令级并行（ILP）支持
- 线程级协作

**示例概念**：
```cpp
// 将 MMA 原子平铺到 128x128x64 的块
using TiledMma = cute::TiledMma<
    MmaAtom,
    Layout<Shape<_16,_8,_4>>,  // Warp 布局：16 warps in M, 8 in N, 4 in K
    Tile<_128,_128,_64>        // Tile 形状
>;
```

**关键概念 - Tiling**：
```
原始矩阵 C = A × B

分块处理：
[C_tile_0] = [A_tile_0] × [B_tile_0]
[C_tile_1] = [A_tile_1] × [B_tile_1]
...
```

---

### 2.3 Collective Layer (集体操作层)

**定义**：时间上的编排，处理流水线管理和同步

**核心组件**：
- `CollectiveMma<>`：协调矩阵乘加的整体流程
- `CollectiveEpilogue<>`：处理输出阶段（后处理、写回）

**功能**：
- **流水线管理**：Global Memory → Shared Memory → Register 的多级流水线
- **同步控制**：线程块内的同步、异步操作的等待
- **资源调度**：Shared Memory 的双缓冲/多缓冲

**流水线示例**：
```
时间轴：
Stage 0: [Load A/B to Shared] → [MMA] → [Store C]
Stage 1:    [Load A/B to Shared] → [MMA] → [Store C]
Stage 2:       [Load A/B to Shared] → [MMA] → [Store C]

重叠：
- Load (Stage i+1) 与 MMA (Stage i) 重叠
- 减少访存延迟
```

**Dispatch Policies (调度策略)**：
- `MainloopSm90TmaGmmaWarpSpecialized`：Hopper 架构的 Warp 专门化主循环
  - 使用 TMA (Tensor Memory Accelerator) 加速访存
  - Warp 专门化：部分 warp 负责数据加载，部分负责计算
- `EpilogueScheduleAuto`：自动选择 Epilogue 调度策略

---

### 2.4 Kernel Layer (内核层)

**定义**：扩展 Collective 操作到整个 Grid 维度

**核心组件**：
- `cutlass::gemm::kernel::GemmUniversal<>`

**功能**：
- 管理多个 CTA (Thread Block) 的协作
- Tile Scheduling：将矩阵分块分配给不同的 CTA

**Tile Scheduling 策略**：

1. **Per-tile CTA (基础调度)**：
```
矩阵 C 分为 M×N 个 tile
每个 CTA 处理一个 tile
简单但可能负载不均衡
```

2. **Persistent Schedulers (持久调度器)**：
```
CTA 不退出，持续获取新任务
适合小问题、高并发场景
减少 kernel 启动开销
```

3. **Stream-K Scheduling**：
```
将 K 维度也进行切分
提高负载均衡
特别适合 M、N 不是 tile 大小整数倍的情况
```

---

### 2.5 Device Layer (设备层)

**定义**：主机端接口，用户最常使用的层级

**核心组件**：
- `cutlass::gemm::device::GemmUniversalAdapter<>`

**功能**：
- 提供简洁的 API
- 管理 CUDA 资源（Grid/Block 配置、Shared Memory 大小）
- 参数验证和错误处理

**使用示例**（伪代码）：
```cpp
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
    cutlass::gemm::kernel::GemmUniversal<
        CollectiveMma,
        CollectiveEpilogue
    >
>;

Gemm gemm_op;
gemm_op.initialize(args);
gemm_op.run(stream);
```

---

## 3. CollectiveBuilder：简化配置

### 3.1 问题背景

直接配置底层抽象需要指定：
- Atom 类型
- Tiled MMA 布局
- 流水线阶段数
- Shared Memory 布局
- ...

这对用户不友好，且容易出错。

### 3.2 CollectiveBuilder 的作用

**CollectiveBuilder** 允许用户通过高层参数指定 GEMM，自动推导底层配置。

**高层参数**：
- 硬件架构（如 `sm_90`）
- 数据类型（如 `half`, `float`, `bfloat16`）
- Tile 形状（如 `128x128x64`）
- 调度策略（如 `WarpSpecialized`）

**自动推导**：
- 最优的 Atom 选择
- Shared Memory 布局
- 流水线深度
- 线程布局

### 3.3 使用示例

```cpp
using CollectiveMma = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,              // 架构：Hopper
    cutlass::arch::OpClassTensorOp,   // 操作类：Tensor Core
    half_t,                            // A 矩阵类型
    cutlass::layout::RowMajor,         // A 布局
    8,                                 // A 对齐
    half_t,                            // B 矩阵类型
    cutlass::layout::ColumnMajor,      // B 布局
    8,                                 // B 对齐
    float,                             // C/D 矩阵类型
    Shape<_128, _128, _64>,            // Tile 形状
    Shape<_2, _1, _1>,                 // Cluster 形状
    cutlass::gemm::KernelTmaWarpSpecialized  // Kernel 调度策略
>::CollectiveOp;
```

---

## 4. 关键技术深入

### 4.1 Warp Specialization (Warp 专门化)

**概念**：将一个 CTA 内的 warp 分为不同角色。

**典型角色**：
- **Producer Warp**：负责数据加载（Global → Shared）
- **Consumer Warp**：负责计算（Shared → Register → MMA）

**优势**：
- 提高指令级并行
- 减少同步开销
- 更好的资源利用（某些 warp 可专注于访存，某些专注于计算）

**示例场景**（Hopper TMA）：
```
Warp 0-1: Producer，使用 TMA 异步加载 A、B 到 Shared Memory
Warp 2-7: Consumer，从 Shared Memory 读取数据并执行 MMA 指令

Pipeline:
- Producer 提前若干 stage 加载数据
- Consumer 等待数据就绪后计算
- 通过异步屏障（Async Barrier）同步
```

### 4.2 TMA (Tensor Memory Accelerator)

**定义**：Hopper 架构引入的硬件加速单元，用于高效的多维数据传输。

**特点**：
- 硬件支持的异步拷贝
- 自动处理复杂的内存布局
- 减少寄存器压力（不需要线程参与）

**PTX 指令示例**：
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [dest], [src, coords], [mbar];
```

**在 CUTLASS 中的使用**：
```cpp
cute::Copy_Atom<SM90_TMA_LOAD_2D>  // 2D TMA 加载
```

### 4.3 Async Pipeline (异步流水线)

**核心思想**：隐藏访存延迟。

**实现方式**：
1. 使用多个 Shared Memory 缓冲区（如 3-stage pipeline）
2. 异步发起内存加载请求
3. 在等待期间执行其他计算
4. 通过屏障（barrier）同步

**代码模式**：
```cpp
// Pseudocode
pipeline.producer_acquire(stage);  // 获取缓冲区
async_copy(gmem_ptr, smem_ptr);    // 异步拷贝
pipeline.producer_commit(stage);   // 提交

pipeline.consumer_wait(stage);     // 等待数据就绪
mma(smem_ptr);                     // 计算
pipeline.consumer_release(stage);  // 释放缓冲区
```

---

## 5. 实践示例

### 5.1 构建一个简单的 FP16 GEMM

```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/collective/collective_builder.hpp>

// Step 1: 定义问题规模和数据类型
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;

// Step 2: 使用 CollectiveBuilder 构建 CollectiveMma
using CollectiveMma = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, cutlass::layout::RowMajor, 8,
    ElementB, cutlass::layout::ColumnMajor, 8,
    ElementAccumulator,
    Shape<_128, _128, _64>,  // CTA Tile
    Shape<_2, _1, _1>,       // Cluster Shape
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename cutlass::arch::ClusterShape<_2, _1, _1>)
    >,
    cutlass::gemm::KernelTmaWarpSpecialized
>::CollectiveOp;

// Step 3: 配置 Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Shape<_128, _128, _64>,
    Shape<_2, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementC,
    ElementC, cutlass::layout::RowMajor, 8,
    ElementC, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Step 4: 组装 Kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMma,
    CollectiveEpilogue
>;

// Step 5: Device Adapter
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Step 6: 运行
void run_gemm(int M, int N, int K) {
    Gemm gemm_op;

    // 准备数据和参数
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    // ... 分配内存、初始化数据 ...

    // 初始化
    typename Gemm::Arguments args{problem_size, /* ... */};
    gemm_op.initialize(args);

    // 执行
    gemm_op.run();
}
```

### 5.2 自定义 Epilogue Fusion

CUTLASS 3.x 支持融合后处理操作（如 bias、activation）：

```cpp
// 示例：融合 bias 和 ReLU
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementC
>;
```

---

## 6. 架构对应关系

### 6.1 GPU 架构与 Compute Capability

| GPU 型号 | 架构 | Compute Capability | CUTLASS 标记 |
|---------|------|-------------------|-------------|
| A100 | Ampere | sm_80 | `cutlass::arch::Sm80` |
| RTX 3060 | Ampere | sm_86 | `cutlass::arch::Sm86` |
| Orin | Ampere | sm_87 | `cutlass::arch::Sm87` |
| RTX 4060 | Ada Lovelace | sm_89 | `cutlass::arch::Sm89` |
| H100 | Hopper | sm_90 | `cutlass::arch::Sm90` |
| B100/B200 | Blackwell | sm_90a | `cutlass::arch::Sm90a` |

### 6.2 硬件特性映射

**Ampere (sm_80/86/87)**：
- Tensor Core 第三代
- Async Copy 支持（`cp.async`）
- Warp Group 级别的 MMA

**Hopper (sm_90)**：
- Tensor Core 第四代
- TMA (Tensor Memory Accelerator)
- Thread Block Cluster
- Warp Specialization 硬件优化

**Blackwell (sm_90a)**：
- 第五代 Tensor Core
- 增强的 TMA 能力
- 更大的 Shared Memory
- 改进的 Warp Specialization

---

## 7. 优化策略

### 7.1 Tile Size 选择

**原则**：
- 平衡计算和访存
- 考虑 Shared Memory 限制
- 考虑寄存器压力

**常见配置**：
- **小问题**：`64x64x32`
- **中等问题**：`128x128x64`
- **大问题**：`256x128x64` 或 `128x256x64`

### 7.2 Pipeline Stages

**深度选择**：
- **2-stage**：适合 Shared Memory 受限场景
- **3-stage**：常见配置，较好的延迟隐藏
- **4+ stage**：适合高延迟访存，需要更多 Shared Memory

### 7.3 Cluster Shape

Hopper 引入了 Thread Block Cluster，允许多个 CTA 共享 L2 缓存和某些资源。

**配置**：
- `Shape<_1, _1, _1>`：单个 CTA（兼容旧架构）
- `Shape<_2, _1, _1>`：2 个 CTA 集群（常用）
- `Shape<_2, _2, _1>`：4 个 CTA 集群（更强协作）

---

## 8. 调试和性能分析

### 8.1 使用 Nsight Compute

```bash
ncu --set full --target-processes all ./gemm_test
```

**关键指标**：
- **SM Efficiency**：流多处理器利用率
- **Memory Throughput**：内存带宽利用率
- **Tensor Core Utilization**：Tensor Core 利用率

### 8.2 CUTLASS Profiler

CUTLASS 提供了内置的性能分析工具：

```bash
./tools/profiler/cutlass_profiler \
    --operation=Gemm \
    --m=4096 --n=4096 --k=4096 \
    --kernels=cutlass_sm90_*
```

---

## 9. 总结

CUTLASS 3.x 的核心价值：

1. **正交抽象**：各层独立，可自由组合
2. **硬件映射**：从指令到设备的清晰层次
3. **自动优化**：CollectiveBuilder 简化配置
4. **未来兼容**：易于扩展到新架构

**学习路径建议**：
1. 理解 GEMM 基本原理
2. 熟悉 CUDA 编程模型
3. 学习 CuTe（CUTLASS 的 Layout 抽象库）
4. 使用 CollectiveBuilder 构建简单 GEMM
5. 深入 Collective 层实现自定义优化
6. 研究特定架构的 Atom 实现

---

## 参考资源

- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS 3.x 文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [CuTe 教程](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)
- [NVIDIA 博客原文](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)

---

**版本信息**：
- CUTLASS 版本：3.x
- 支持架构：Ampere (sm_80+), Hopper (sm_90), Blackwell (sm_100)
- 最后更新：2026-01
