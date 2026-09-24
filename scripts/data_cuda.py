# -*- coding: utf-8 -*-
"""
CUDA Kernel Track (12 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

CUDA_LESSONS = {
    "lesson01_vector_add.qmd": {
        "title": "第 1 课：线程映射与 Grid-Stride Loop",
        "fig1": """
```mermaid
flowchart TD
    Grid["Grid 网格 (一维/多维)"] --> Block0["Block 0"]
    Grid --> Block1["Block 1"]
    Grid --> BlockN["Block (gridDim.x - 1)"]
    Block0 --> Warp0["Warp 0 (线程 0~31)"]
    Block0 --> Warp1["Warp 1 (线程 32~63)"]
    Warp0 --> T0["Thread 0 访问 a[0]"]
    Warp0 --> T1["Thread 1 访问 a[1]"]
    Warp0 --> T31["Thread 31 访问 a[31]"]
    Note["连续线程访问连续内存地址 ➔ 128字节合并访存 (Coalesced Access)"]
```
<p class="caption" align="center"><em>图 1-1：GPU 线程层次（Grid/Block/Warp/Thread）与合并访存映射架构</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Start["线程启动: idx = blockIdx.x * blockDim.x + threadIdx.x"] --> Cond{"idx < N ?"}
    Cond -->|是| Calc["执行计算: c[idx] = a[idx] + b[idx]"]
    Calc --> Stride["跨步前进: idx += blockDim.x * gridDim.x"]
    Stride --> Cond
    Cond -->|否| End["线程退出 (自然处理任意长度 N 与尾块)"]
```
<p class="caption" align="center"><em>图 1-2：Grid-Stride Loop 网格跨步循环执行控制流</em></p>
"""
    },
    "lesson02_reduce_sum.qmd": {
        "title": "第 2 课：Warp/Block 归约与两阶段求和",
        "fig1": """
::: {.img-card}
![](assets/figs/cuda_warp_shuffle_latency.png){width="80%"}
<p class="caption">图 2-1：Warp Shuffle 与 Shared Memory 两阶段规约时延对比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant T as Block 内各 Warp 线程
    participant S as 共享内存 (Shared Memory)
    participant W0 as Warp 0 (主归约 Warp)
    participant G as 全局内存 (GMEM Output)
    
    T->>T: 各 Warp 独立执行 __shfl_down_sync 归约
    T->>S: 各 Warp 的 Lane 0 将局部和存入 SMem[warp_id]
    Note over T,S: __syncthreads() 保证写完成
    S->>W0: Warp 0 读出各 Warp 局部和
    W0->>W0: Warp 0 再次执行 Shuffle 归约
    W0->>G: Block 内 Lane 0 原子累加或直接写出最终标量
```
<p class="caption" align="center"><em>图 2-2：Block 级两阶段求和（Warp 内洗牌 + 跨 Warp 归约）执行时序流</em></p>
"""
    },
    "lesson03_gemv.qmd": {
        "title": "第 3 课：GEMV 与行级并行",
        "fig1": """
```mermaid
flowchart LR
    subgraph Matrix["矩阵 A (M × K)"]
        Row0["Row 0 ➔ 分配给 Warp 0"]
        Row1["Row 1 ➔ 分配给 Warp 1"]
        RowM["Row M ➔ 分配给 Warp M"]
    end
    subgraph Vector["向量 x (K × 1)"]
        Vx["常驻 L1/L2 Cache 广播读取"]
    end
    subgraph Out["输出 y (M × 1)"]
        Y0["y[0]"]
        Y1["y[1]"]
        YM["y[M]"]
    end
    Row0 & Vx --> Y0
    Row1 & Vx --> Y1
    RowM & Vx --> YM
```
<p class="caption" align="center"><em>图 3-1：GEMV 矩阵-向量乘行级映射与数据复用拓扑</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Init["Warp 绑定矩阵第 row 行"] --> Loop["Warp 内各线程跨步读取 A[row, c] 并与 x[c] 做乘积累加"]
    Loop --> Reduce["Warp 内执行 Shuffle 求和归约"]
    Reduce --> Write["Lane 0 将点积和加上偏置写回全局内存 y[row]"]
```
<p class="caption" align="center"><em>图 3-2：GEMV 单行点积累加与 Warp 归约执行流水</em></p>
"""
    },
    "lesson04_rmsnorm.qmd": {
        "title": "第 4 课：RMSNorm：统计量与两遍访问",
        "fig1": """
```mermaid
flowchart LR
    subgraph RMSNormMath["RMSNorm 核心计算链路"]
        direction TB
        X["输入向量 x"] --> Square["平方累加: sum(x_i^2)"]
        Square --> Mean["均方根: RMS = sqrt(mean(x^2) + eps)"]
        Mean --> Scale["归一化并仿射: y_i = (x_i / RMS) * weight_i"]
    end
```
<p class="caption" align="center"><em>图 4-1：RMSNorm 均方根归一化与缩放系数计算图</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant T as 线程束 (Threads)
    participant S as 共享内存 (SMem)
    participant GM as 全局内存 (HBM)
    
    Note over T,GM: 第一遍遍历 (Pass 1: 计算平方和)
    T->>GM: 读取 x 并累加本地局部平方和
    T->>S: 块内归约得出整行平方均值，计算 1 / RMS
    Note over T,S: 同步屏障 __syncthreads()
    Note over T,GM: 第二遍遍历 (Pass 2: 归一化与写回)
    T->>GM: 重新加载 x[i]，乘以 1/RMS 与 weight[i]
    T->>GM: 将归一化结果写回全局内存 y[i]
```
<p class="caption" align="center"><em>图 4-2：RMSNorm 两阶段访存扫描与片上归约时序流程</em></p>
"""
    },
    "lesson05_layernorm.qmd": {
        "title": "第 5 课：LayerNorm：均值、方差与数值稳定",
        "fig1": """
```mermaid
flowchart TD
    subgraph Welford["Welford 在线单遍递推算法"]
        Old["上一状态: (count, mean, M2)"] --> In["新样本 x"]
        In --> Delta["delta = x - mean"]
        Delta --> UpdateMean["mean_new = mean + delta / (count + 1)"]
        UpdateMean --> UpdateM2["M2_new = M2 + delta * (x - mean_new)"]
        UpdateM2 --> Var["方差 var = M2 / N"]
    end
```
<p class="caption" align="center"><em>图 5-1：Welford 在线数值稳定递推均值与方差架构</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Read["读入输入 x"] --> WelfordReduce["Warp & Block 级 Welford 结合律树状归约"]
    WelfordReduce --> Broadcast["广播均值 mu 与标准差 1/sqrt(var + eps)"]
    Broadcast --> Normalize["执行 (x - mu) * rstd * gamma + beta 写回"]
```
<p class="caption" align="center"><em>图 5-2：LayerNorm 线程级统计量归约与输出写回流程</em></p>
"""
    },
    "lesson06_softmax_naive.qmd": {
        "title": "第 6 课：稳定 Softmax 与朴素复杂度",
        "fig1": """
```mermaid
flowchart LR
    subgraph StableSoftmax["安全防溢出 Softmax 技巧"]
        direction TB
        Raw["原始分数 x_i (可能极大)"] --> Max["求全局最大值 m = max(x_i)"]
        Max --> Shift["平移指数: exp(x_i - m) (指数区间 <= 0，绝不溢出)"]
        Shift --> Sum["累加和: d = sum(exp(x_i - m))"]
        Sum --> Div["归一化输出: y_i = exp(x_i - m) / d"]
    end
```
<p class="caption" align="center"><em>图 6-1：数值安全 Softmax 减最大值平移与归一化机制</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant T as 线程集合
    participant S as 共享内存 (SMem)
    participant G as 全局内存 (GMEM)
    
    T->>G: 第一次遍历: 读取 x[i]，归约求全局最大值 m
    T->>S: 广播最大值 m
    T->>G: 第二次遍历: 重新读取 x[i]，计算 exp(x[i]-m) 并累加总和 d
    T->>S: 广播分母 d
    T->>G: 第三次遍历: 计算最终概率值并写回全局内存
```
<p class="caption" align="center"><em>图 6-2：朴素 Softmax 三遍访存流水线与内存开销瓶颈</em></p>
"""
    },
    "lesson07_online_softmax.qmd": {
        "title": "第 7 课：Online Softmax 与可结合状态",
        "fig1": """
```mermaid
flowchart LR
    Input["输入分块 x_i"] --> Max["更新局部最大值: m_new = max(m_old, x_i)"]
    Max --> Scale["修正历史累加和: d_new = d_old * e^(m_old - m_new) + e^(x_i - m_new)"]
    Scale --> Flash["单遍扫描 (1-Pass) 无需显存常驻完整矩阵 ➔ FlashAttention 底层基石"]
```
<p class="caption" align="center"><em>图 7-1：Online Softmax 可结合状态增量递推配平机制</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Init["初始化状态: m = -INF, d = 0, O = 0"] --> Load["分块加载下一个 Tile 数据"]
    Load --> UpdateM["计算当前块的最大值并比较更新全局 m"]
    UpdateM --> Rescale["乘上因子 exp(m_prev - m_new) 修正此前积累的输出矩阵 O"]
    Rescale --> Acc["将当前分块的注意力结果追加到 O 中"]
    Acc --> Check{"是否有剩余分块?"}
    Check -->|是| Load
    Check -->|否| Final["除以最终累加和 d 得到准确输出，零中间 HBM 写回"]
```
<p class="caption" align="center"><em>图 7-2：单遍扫描 Online Softmax 增量更新与缩放流程</em></p>
"""
    },
    "lesson08_transpose_naive.qmd": {
        "title": "第 8 课：朴素转置与合并访存",
        "fig1": """
```mermaid
flowchart TD
    subgraph Contrast["转置访存冲突对比"]
        subgraph Read["读操作 (合并访存)"]
            T_read["相邻线程 threadIdx.x 访问相邻列 A[y, x] ➔ 完美 128B DRAM 突发传输"]
        end
        subgraph Write["写操作 (跨步访存)"]
            T_write["相邻线程写相邻行 B[x, y] ➔ 跨步跨越整行步长，触发 32 次独立 32B 事务 (吞吐跌至 1/8)"]
        end
    end
```
<p class="caption" align="center"><em>图 8-1：二维矩阵转置合并读与跨步写访存模式冲突图</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Coalesced["输入内存读取: 合并访存利用率 100%"] --> DirectWrite["直接写入转置位置: 内存控制器被散落写入击穿"]
    DirectWrite --> Bottleneck["显存控制器总线效率严重低下 ➔ 必须引入共享内存缓冲"]
```
<p class="caption" align="center"><em>图 8-2：朴素转置内存事务利用率低下导致性能腰斩的因果流</em></p>
"""
    },
    "lesson09_transpose_tiled.qmd": {
        "title": "第 9 课：共享内存转置与 Bank Conflict",
        "fig1": """
::: {.img-card}
![](assets/figs/cuda_smem_bank_conflict.png){width="85%"}
<p class="caption">图 9-1：Shared Memory 跨步冲突对有效吞吐的影响及填充消除原理（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant GM_In as 全局内存输入 A
    participant SM as 共享内存 tile[32][33]
    participant GM_Out as 全局内存输出 B
    
    GM_In->>SM: 线程合并读取 A[y, x] 并写入本地 tile[ty][tx]
    Note over SM: __syncthreads() 保证整个 Tile 填满
    SM->>GM_Out: 线程按转置坐标合并读取 tile[tx][ty]，并合并写入 B[x, y]
    Note over GM_In,GM_Out: 读写双向全部达成合并访存且 SMem 无 Bank 冲突
```
<p class="caption" align="center"><em>图 9-2：基于共享内存中继的双向合并转置执行流</em></p>
"""
    },
    "lesson10_gemm_tiled.qmd": {
        "title": "第 10 课：共享内存 Tiled GEMM",
        "fig1": """
::: {.img-card}
![](assets/figs/cuda_roofline.png){width="85%"}
<p class="caption">图 10-1：NVIDIA H100 Roofline 性能模型与分块 GEMM 算力/访存比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    LoadTile["从全局显存加载下一个 A/B 瓦片到 SMem"] --> Sync1["__syncthreads()"]
    Sync1 --> Compute["各线程从 SMem 读入寄存器，执行外积/乘加累加"]
    Compute --> Sync2["__syncthreads()"]
    Sync2 --> NextK{"K 维是否结束?"}
    NextK -->|否| LoadTile
    NextK -->|是| WriteOut["将寄存器 C 结果合并写回全局显存"]
```
<p class="caption" align="center"><em>图 10-2：分块矩阵乘法双层同步与步进累加流程图</em></p>
"""
    },
    "lesson11_silu.qmd": {
        "title": "第 11 课：SiLU 与算子融合",
        "fig1": """
```mermaid
flowchart LR
    subgraph Separate["非融合算子: 3 次往返全局显存"]
        direction TB
        X1["X (GMEM)"] --> Sig["Sigmoid Kernel"] --> S_out["Sigmoid(X) 写回 GMEM"]
        S_out --> Mul["Multiply Kernel"] --> Y1["Y = X * Sigmoid(X) 写回 GMEM"]
    end
    subgraph Fused["融合算子 (Kernel Fusion): 单次访存"]
        direction TB
        X2["X (GMEM)"] --> Load["一次加载到寄存器"] --> Math["val / (1.0f + expf(-val))"] --> Y2["写回 Y (GMEM)"]
    end
```
<p class="caption" align="center"><em>图 11-1：非融合往返显存带宽瓶颈 vs 融合算子寄存器直出对比</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    In["输入全局内存指针与尺寸 N"] --> Launch["单网格 launch 激活融合 Kernel"]
    Launch --> Reg["float4 向量化加载 4 个 float 进寄存器"]
    Reg --> Comp["单指令高速执行数学级联计算"]
    Comp --> Store["float4 向量化写回全局内存，节省 66% 访存带宽"]
```
<p class="caption" align="center"><em>图 11-2：SiLU 激活算子向量化融合流水线</em></p>
"""
    },
    "lesson12_argmax.qmd": {
        "title": "第 12 课：ArgMax 对归约与确定性",
        "fig1": """
```mermaid
flowchart LR
    subgraph Pair["键值对元组结构"]
        Val["val: 浮点极值 (float)"]
        Idx["idx: 整数全局下标 (int)"]
    end
    Pair --> Logic{"比较条件: val_b > val_a 或 (val_b == val_a 且 idx_b < idx_a)"}
    Logic --> Winner["胜者晋级 ➔ 保证 Tie-breaking 严格确定性"]
```
<p class="caption" align="center"><em>图 12-1：ArgMax 键值对元组结构与确定性仲裁规则</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant L as Warp 各线程 (Lane 0~31)
    participant R as __shfl_down_sync 归约
    participant Winner as 最终极值与最小下标
    
    L->>R: 各 Lane 打包 (val, idx) 参与两两比较
    R->>R: 16 -> 8 -> 4 -> 2 -> 1 步进折半比较
    R->>Winner: Lane 0 获得该 Warp 内最大数值及最小下标
```
<p class="caption" align="center"><em>图 12-2：Warp 级确定性 ArgMax 归约时序流</em></p>
"""
    }
}
