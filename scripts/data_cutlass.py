# -*- coding: utf-8 -*-
"""
CUTLASS / CuTe Track (8 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

CUTLASS_LESSONS = {
    "lesson01_hierarchy.qmd": {
        "title": "第 1 课：GPU 层级、Tensor Core 与算术强度",
        "fig1": """
::: {.img-card}
![SM 架构演进图](assets/figs/fig_02_从_Volta_到_Blackwell_的_SM_架构演进.png){width="90%"}
<p class="caption">图 1-1：从 Volta 到 Blackwell 的 NVIDIA GPU SM 架构演进与 Tensor Core 升级</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    subgraph MemHierarchy["GEMM 算术强度三级访存层次"]
        GMEM["全局显存 HBM: 容量大 (80GB+), 带宽约 2~3 TB/s"] -->|加载分块 Tile| SMEM["共享内存 SMem: 容量中 (200KB+/SM), 带宽约 15~20 TB/s"]
        SMEM -->|加载矩阵片段 Fragment| RF["寄存器文件 RF: 容量小 (~64K 寄存器/SM), 带宽高达 30+ TB/s"]
        RF --> TC["Tensor Core MMA 矩阵计算单元 (最高达到 PFLOPS)"]
    end
```
<p class="caption" align="center"><em>图 1-2：算术强度在 GPU 存储金字塔（GMEM-SMem-RF）中的分层复用流</em></p>
"""
    },
    "lesson02_layout.qmd": {
        "title": "第 2 课：CuTe Layout：Shape、Stride 与坐标映射",
        "fig1": """
::: {.img-card}
![CuTe Layout 映射图](assets/figs/fig_01_CuTe_的第_1_种_Layout____Tensor_Layout.png){width="90%"}
<p class="caption">图 2-1：CuTe Layout 核心代数：Shape、Stride 与多维逻辑坐标到一维内存物理偏移的映射</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    Coord["逻辑坐标 (m, n)"] --> InnerProduct["坐标与步长做代数内积: offset = m * stride_m + n * stride_n"]
    InnerProduct --> MemOffset["物理内存一维下标 1D Pointer Offset"]
    MemOffset --> Memory["高效定位全局显存或共享内存单元"]
```
<p class="caption" align="center"><em>图 2-2：CuTe 核心代数坐标变换与内存地址计算流水</em></p>
"""
    },
    "lesson03_tensor_partition.qmd": {
        "title": "第 3 课：CuTe Tensor、视图与线程分区",
        "fig1": """
::: {.img-card}
![make_tensor 视图图](assets/figs/fig_03_CuTe_API____make_tensor.png){width="90%"}
<p class="caption">图 3-1：CuTe API make_tensor 与内存抽象视图</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    GlobalTensor["全局张量 Tensor (由指针与 Layout 构成)"] --> Partition["CuTe local_partition(tensor, thr_layout, thread_idx)"]
    Partition --> ThreadLocalTensor["每个线程持有的局部切片视图 Tensor"]
    ThreadLocalTensor --> Math["线程无锁并行执行局部计算"]
```
<p class="caption" align="center"><em>图 3-2：CuTe 张量线程分区（Thread Partitioning）与数据切分流</em></p>
"""
    },
    "lesson04_tiled_copy.qmd": {
        "title": "第 4 课：Tiled Copy、向量化与 Predication",
        "fig1": """
::: {.img-card}
![Tiled Copy API 图](assets/figs/fig_05_make_tiled_copy_API.png){width="90%"}
<p class="caption">图 4-1：CUTLASS Tiled Copy 构造与数据搬运抽象</p>
:::
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant T as 线程集
    participant Pred as 边界掩码 (Predication)
    participant HW as cp.async 硬件引擎
    participant SM as 共享内存 (SMem)
    
    T->>Pred: 判定坐标是否位于张量有效边界内
    alt 边界内
        Pred->>HW: 发射 128 位宽向量化 cp.async 异步拷贝指令
        HW->>SM: 硬件直通写入共享内存 (无需经过寄存器)
    else 越界
        Pred->>SM: 写入 0 (Padding 零填充保护)
    end
```
<p class="caption" align="center"><em>图 4-2：Tiled Copy 向量化加载与掩码谓词保护执行时序</em></p>
"""
    },
    "lesson05_tiled_mma.qmd": {
        "title": "第 5 课：MMA Atom、Tiled MMA 与三级 Tiling",
        "fig1": """
::: {.img-card}
![Tiled MMA API 图解](assets/figs/fig_08_make_tiled_mma_API_图解.png){width="90%"}
<p class="caption">图 5-1：CUTLASS make_tiled_mma API 与硬件指令对应关系图解</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    subgraph ThreeTier["GEMM 三级 Tiling 层次"]
        CTATile["1. CTA 级 Tiling: 分块置于共享内存 (如 128×128)"]
        WarpTile["2. Warp 级 Tiling: 线程束分块置于寄存器 (如 64×32)"]
        MMAAtom["3. MMA Atom 硬件原语: 底层 mma.sync 指令 (如 16×8×16)"]
        CTATile --> WarpTile --> MMAAtom
    end
```
<p class="caption" align="center"><em>图 5-2：从硬件 Atom 到 Thread Block 的三级分层 GEMM 构建流</em></p>
"""
    },
    "lesson06_smem_pipeline.qmd": {
        "title": "第 6 课：Shared Memory 布局与异步流水",
        "fig1": """
::: {.img-card}
![异步拷贝流水图](assets/figs/fig_06_cp_async_拷贝原理_图源自_NVIDIA_Ampere_白皮书_.png){width="90%"}
<p class="caption">图 6-1：cp.async 异步拷贝原理（图源自 NVIDIA Ampere 白皮书）</p>
:::
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant Stage0 as SMem Stage 0
    participant Stage1 as SMem Stage 1
    participant MMA as Tensor Core 计算
    
    Note over Stage0,MMA: 异步多阶段双缓冲/多缓冲流水
    Stage0->>Stage0: 异步加载第 k 步数据 (cp.async)
    Stage0->>MMA: 读取第 k 步数据并执行矩阵乘
    Stage1->>Stage1: 同时异步预取第 k+1 步数据
    MMA-->>Stage0: 计算完毕释放 Stage 0
    Stage1->>MMA: 立即计算第 k+1 步，计算完全隐藏数据传输
```
<p class="caption" align="center"><em>图 6-2：Ampere/Hopper 多级异步环形流水线（Circular Pipeline）执行流</em></p>
"""
    },
    "lesson07_mixed_precision.qmd": {
        "title": "第 7 课：混合精度、Accumulator 与 Epilogue",
        "fig1": """
::: {.img-card}
![FP8 精度计算图](assets/figs/fig_02_DeepSeek_V3_中的_FP8_精度计算.png){width="90%"}
<p class="caption">图 7-1：DeepSeek-V3 / 现代大模型中的 FP8 精度计算与分块缩放机制</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    Inputs["低精度输入 (FP8 / BF16)"] --> MMA["Tensor Core 高速矩阵乘"]
    MMA --> Accum["FP32 高精度累加器 (防止下溢与累加误差)"]
    Accum --> Epilogue["Epilogue 算子融合 (Bias + ReLU/SiLU + 量化缩放)"]
    Epilogue --> Out["写回目标格式到全局显存"]
```
<p class="caption" align="center"><em>图 7-2：CUTLASS 混合精度累加与 Epilogue 融合输出流水线</em></p>
"""
    },
    "lesson08_kernel_selection.qmd": {
        "title": "第 8 课：Kernel 组合、Dispatch 与 Profiling",
        "fig1": """
::: {.img-card}
![Nsight Compute 概览](assets/figs/fig_08_Nsight_Compute_概览界面.png){width="90%"}
<p class="caption">图 8-1：Nsight Compute (NCU) 算子性能分析概览界面</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    Shape["输入 GEMM 形状 (M, N, K)"] --> Heuristic["启发式规则 / 预调优表格"]
    Heuristic --> Config["选定最优 Tile 尺寸与多级流水级数"]
    Config --> Dispatch["C++ 模板特化 Kernel Dispatch"]
    Dispatch --> Run["GPU 高性能并发执行"]
```
<p class="caption" align="center"><em>图 8-2：CUTLASS Kernel 动态配置选型与分发调度流程</em></p>
"""
    }
}
