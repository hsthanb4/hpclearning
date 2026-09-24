# -*- coding: utf-8 -*-
"""
Triton Kernel Track (10 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

TRITON_LESSONS = {
    "lesson01_fill.qmd": {
        "title": "第 1 课：Program、Block 与 Mask",
        "fig1": """
```mermaid
flowchart TD
    Grid["1D Program 网格 (tl.program_id(0))"] --> Prog0["Program 0: 处理 [0, BLOCK_SIZE)"]
    Grid --> Prog1["Program 1: 处理 [BLOCK_SIZE, 2*BLOCK_SIZE)"]
    Grid --> ProgK["Program k: 处理 [k*BLOCK, (k+1)*BLOCK)"]
    ProgK --> Offset["偏移计算: offsets = k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)"]
```
<p class="caption" align="center"><em>图 1-1：Triton SPMD Block 划分与块偏移指针代数</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Offsets["计算逻辑元素索引 offsets"] --> Mask["生成布尔掩码: mask = offsets < N"]
    Mask --> Load["安全加载/存储: tl.store(ptr + offsets, val, mask=mask)"]
    Load --> Out["边界外非法越界被自动抑制 (无任何非法访存)"]
```
<p class="caption" align="center"><em>图 1-2：Triton tl.store 边界掩码 (Mask) 保护执行流程图</em></p>
"""
    },
    "lesson02_vector_add.qmd": {
        "title": "第 2 课：Load/Store 与向量加法",
        "fig1": """
```mermaid
flowchart LR
    PtrA["指针 a_ptr + offsets"] -->|tl.load| RegA["向量寄存器块 a"]
    PtrB["指针 b_ptr + offsets"] -->|tl.load| RegB["向量寄存器块 b"]
    RegA & RegB -->|元素级 + 运算| RegC["结果向量块 c"]
    RegC -->|tl.store| PtrC["写回 c_ptr + offsets"]
```
<p class="caption" align="center"><em>图 2-1：Triton 向量块级加载、计算与写回数据通路</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Host["Python Host: launch vector_add[(n_blocks,)](a, b, c, n)"] --> Grid["GPU JIT 编译并根据 BLOCK_SIZE 分发网格"]
    Grid --> Compute["各 Program 块并行读取与执行 SIMD 加法"]
    Compute --> Sync["无全局同步，独立完成并退出"]
```
<p class="caption" align="center"><em>图 2-2：Triton 向量加法端到端发射与并行执行流程</em></p>
"""
    },
    "lesson03_relu.qmd": {
        "title": "第 3 课：逐元素激活与融合边界",
        "fig1": """
```mermaid
flowchart LR
    subgraph ActCompare["逐元素激活函数映射"]
        X["输入 x"] --> ReLU["tl.maximum(x, 0) ➔ 截断负数"]
        X --> GeLU["tl.where(x > 0, x, x * alpha) ➔ 连续平滑"]
    end
```
<p class="caption" align="center"><em>图 3-1：Triton 逐元素激活函数数学映射比较</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    In["加载特征 x"] --> Bias["加偏置 (+ bias)"]
    Bias --> Act["应用激活 (ReLU / SiLU)"]
    Act --> Dropout["条件丢弃 (Dropout Mask)"]
    Dropout --> Store["单次写回全局显存 ➔ 节约 75% 访存带宽"]
```
<p class="caption" align="center"><em>图 3-2：Triton 算子深度融合流水线</em></p>
"""
    },
    "lesson04_add_bias.qmd": {
        "title": "第 4 课：二维 Grid、Stride 与广播",
        "fig1": """
```mermaid
flowchart TD
    subgraph Stride2D["二维张量行优先 Stride 映射"]
        Coord["二维坐标 (row, col)"] --> Formula["一维物理下标 = row * stride_row + col * stride_col"]
        Formula --> Mem["连续跨步读取二维矩阵块"]
    end
```
<p class="caption" align="center"><em>图 4-1：Triton 二维 Block 与行优先 Stride 计算代数</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Bias["1D 偏置 (BLOCK_N,)"] --> Expand["切片扩展维度 [None, :] 变成 (1, BLOCK_N)"]
    Expand --> Broad["沿行方向自动广播与 (BLOCK_M, BLOCK_N) 矩阵相加"]
```
<p class="caption" align="center"><em>图 4-2：Triton 广播机制（Broadcasting）执行流</em></p>
"""
    },
    "lesson05_row_sum.qmd": {
        "title": "第 5 课：行归约与 Padding 单位元",
        "fig1": """
```mermaid
flowchart LR
    RowData["矩阵第 row 行数据"] --> MaskPadding["超出真实列数 N 的元素填充单位元 0.0 (加法零元)"]
    MaskPadding --> TLSum["tl.sum(block, axis=1) 沿列求和"]
    TLSum --> Scalar["输出当前行总和标量"]
```
<p class="caption" align="center"><em>图 5-1：Triton 行归约结构与 Padding 加法单位元保护图</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Prog["Program 绑定矩阵单行"] --> Load["分块加载行内数据并应用掩码"]
    Load --> Tree["Triton 片上折半树状规约"]
    Tree --> Store["单值写回输出向量 out[row]"]
```
<p class="caption" align="center"><em>图 5-2：行归约内部树状累加与写回流程</em></p>
"""
    },
    "lesson06_mean_var.qmd": {
        "title": "第 6 课：均值方差与 Mask 二次传播",
        "fig1": """
```mermaid
flowchart LR
    subgraph Stats["统计量计算两阶段"]
        Sum["第一次归约: sum(x) / valid_count ➔ 得到均值 mu"]
        Var["第二次归约: sum((x - mu)^2) / valid_count ➔ 得到方差 var"]
    end
```
<p class="caption" align="center"><em>图 6-1：均值与方差的两阶段掩码统计量结构</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    In["输入块"] --> Pass1["Pass 1: 计算均值并保留 valid_count"]
    Pass1 --> Pass2["Pass 2: 重新遍历差值平方并排除越界元素"]
    Pass2 --> Out["同时输出均值与无偏方差"]
```
<p class="caption" align="center"><em>图 6-2：两阶段掩码二次传播计算时序流</em></p>
"""
    },
    "lesson07_softmax.qmd": {
        "title": "第 7 课：稳定 Softmax 与融合",
        "fig1": """
::: {.img-card}
![](assets/figs/triton_fused_softmax_speedup.png){width="85%"}
<p class="caption">图 7-1：Triton 融合 Softmax 对比 PyTorch 原生算子执行延迟与加速比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    Load["从 HBM 加载行向量到寄存器"] --> LocalMax["求局部最大值并防溢出"]
    LocalMax --> ExpSum["求指数和与分母"]
    ExpSum --> NormWrite["除以分母并写回目标显存"]
```
<p class="caption" align="center"><em>图 7-2：融合 Softmax 一体化执行时序图</em></p>
"""
    },
    "lesson08_transpose.qmd": {
        "title": "第 8 课：二维 Tile 与转置",
        "fig1": """
```mermaid
flowchart LR
    subgraph TileSwap["2D Tile 坐标与指针轴交换"]
        InTile["输入分块 (BLOCK_M, BLOCK_N)"] --> Trans["转置操作 tl.trans(x) ➔ (BLOCK_N, BLOCK_M)"]
        Trans --> OutTile["按列转置写回目标地址"]
    end
```
<p class="caption" align="center"><em>图 8-1：二维分块转置与内存步长重塑架构</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Program2D["2D Program 划分 (pid_m, pid_n)"] --> Load["合并加载源矩阵分块"]
    Load --> SMemTrans["Triton 编译器自动在片上执行无冲突转置"]
    SMemTrans --> Store["合并写回目标转置矩阵"]
```
<p class="caption" align="center"><em>图 8-2：2D 分块转置全局合并读写流程</em></p>
"""
    },
    "lesson09_matmul.qmd": {
        "title": "第 9 课：Blocked Matmul 与 FP32 累加",
        "fig1": """
```mermaid
flowchart TD
    subgraph BlockGEMM["分块矩阵乘 (BLOCK_M × BLOCK_N)"]
        A_k["A 瓦片 (BLOCK_M × BLOCK_K)"]
        B_k["B 瓦片 (BLOCK_K × BLOCK_N)"]
        Acc["累加器 accumulator (FP32 精度)"]
        A_k & B_k ➔|tl.dot(a, b, acc)| Acc
    end
```
<p class="caption" align="center"><em>图 9-1：Blocked Matmul 瓦片乘法与 FP32 累加器保持精度架构</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Init["初始化 acc = zeros((BLOCK_M, BLOCK_N), dtype=fp32)"] --> LoopK["K 维步进: 加载 A_tile 与 B_tile"]
    LoopK --> Mma["acc = tl.dot(A_tile, B_tile, acc)"]
    Mma --> Check{"K 轴遍历完毕?"}
    Check -->|否| LoopK
    Check -->|是| CastStore["转回 FP16/BF16 并写回 C[i, j]"]
```
<p class="caption" align="center"><em>图 9-2：Blocked Matmul K 维分块遍历累加流水线</em></p>
"""
    },
    "lesson10_dropout.qmd": {
        "title": "第 10 课：无状态随机数与 Dropout",
        "fig1": """
```mermaid
flowchart LR
    subgraph PhiloxRNG["Philox 无状态随机数生成器"]
        Seed["随机种子 seed"] & Offset["绝对元素偏移 offset"] --> Philox["tl.rand(seed, offsets)"]
        Philox --> Uniform["输出 [0, 1) 均匀分布随机浮点数"]
    end
```
<p class="caption" align="center"><em>图 10-1：Philox 无状态随机数根据种子与偏移即时生成机制</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Generate["通过 Philox 生成随机数 r"] --> Compare["判定 r > p 生成保留掩码 keep_mask"]
    Compare --> Scale["有效值乘以缩放因子 1 / (1 - p)"]
    Scale --> Store["丢弃项置 0 并写回输出张量"]
```
<p class="caption" align="center"><em>图 10-2：Inverted Dropout 掩码生成与前向缩放执行流</em></p>
"""
    }
}
