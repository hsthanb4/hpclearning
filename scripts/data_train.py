# -*- coding: utf-8 -*-
"""
Training System Track (14 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

TRAIN_LESSONS = {
    "lesson01.qmd": {
        "title": "第 1 课：训练流程与显存账本",
        "fig1": """
::: {.img-card}
![](assets/llm_memory_breakdown.jpg){width="95%"}
<p class="caption">图 1-1：大模型训练显存账本构成（静态参数、梯度、优化器状态与动态激活）</p>
:::
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant F as 前向传播 (Forward)
    participant B as 反向传播 (Backward)
    participant O as 优化器更新 (Optimizer Step)
    participant M as 显存占用 (GPU Memory)
    
    Note over F,M: 典型训练步显存动态生命周期
    F->>M: 申请激活值内存 (随层数累加至峰值)
    Note right of M: 激活峰值达到最高点
    B->>M: 逐层释放激活值，并申请梯度存储空间
    Note right of M: 激活释放与梯度建立交替
    O->>M: 申请 Master Weights 与 FP32 Adam 动量/方差状态 (8N bytes)
    O->>M: 更新参数完毕，清空/归零梯度 (释放 2N/4N bytes)
    Note over F,M: 回到基线静态常驻显存状态
```
<p class="caption" align="center"><em>图 1-2：训练单步生命周期中的显存峰值与状态释放时序流</em></p>
"""
    },
    "lesson02.qmd": {
        "title": "第 2 课：激活、重计算与梯度累积",
        "fig1": """
::: {.img-card}
![](assets/activation_recompute.jpg){width="95%"}
<p class="caption">图 2-1：激活重计算（Activation Checkpointing）机制、显存曲线对比与计算/显存权衡</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    subgraph MicroBatches["梯度累积 Micro-Batch 调度时序"]
        direction TB
        MB1["Micro-Batch 1 (前向+反向)"] -->|累加梯度| G["梯度缓冲区 (Grad Buffer)"]
        MB2["Micro-Batch 2 (前向+反向)"] -->|累加梯度| G
        MBN["Micro-Batch N (前向+反向)"] -->|累加梯度| G
        G --> Step["除以 N 归一化均值 ➔ Optimizer Step"]
    end
```
<p class="caption" align="center"><em>图 2-2：梯度累积跨微批次累加与优化器单步平均执行流</em></p>
"""
    },
    "lesson03.qmd": {
        "title": "第 3 课：分布式通信原语",
        "fig1": """
::: {.img-card}
<img src="assets/allreduce_vs_reducescatter_allgather.jpeg" width="90%" alt="AllReduce vs ReduceScatter + AllGather 对比图解" />
<p class="caption">图 3-1：AllReduce vs ReduceScatter + AllGather 机制与显存/通信对比（本地生图模型生成）</p>
:::
""",
        "fig2": """
::: {.img-card}
<img src="assets/reducescatter_allgather.svg" width="90%" alt="分块状态流转图" />
<p class="caption">图 3-2：Ring 算法各 Rank 分块在 ReduceScatter 与 AllGather 两个阶段的流转状态</p>
:::
"""
    },
    "lesson04.qmd": {
        "title": "第 4 课：数据并行 DP 与 DDP",
        "fig1": """
```mermaid
flowchart TD
    subgraph DataParallel["数据并行架构 (DDP)"]
        Data["全局 Batch 训练数据"] --> Split1["分片 0 (Rank 0)"]
        Data --> Split2["分片 1 (Rank 1)"]
        Data --> Split3["分片 2 (Rank 2)"]
        Split1 --> Model0["卡 0: 模型完整副本 W"]
        Split2 --> Model1["卡 1: 模型完整副本 W"]
        Split3 --> Model2["卡 2: 模型完整副本 W"]
        Model0 --> Grad0["本地梯度 g_0"]
        Model1 --> Grad1["本地梯度 g_1"]
        Model2 --> Grad2["本地梯度 g_2"]
        Grad0 & Grad1 & Grad2 --> AR["Ring AllReduce 全局同步"]
        AR --> Update["各卡同步更新同构参数 W_new"]
    end
```
<p class="caption" align="center"><em>图 4-1：数据并行（DDP）同构副本前向与全量梯度归约架构</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant B as 反向求导 (Backward Compute)
    participant Bucket as 梯度分桶 (Bucket 0 / Bucket 1)
    participant NCCL as 异步通信流 (Ring AllReduce)
    Note over B,NCCL: PyTorch DDP 计算与通信重叠流水线
    B->>Bucket: 逐层反向求导 (Layer N)
    Bucket-->>Bucket: 梯度写入，达到阈值 (如 25MB)
    Bucket->>NCCL: 异步触发 AllReduce (Bucket 0)
    B->>Bucket: 继续求导上一层 (Layer N-1, N-2)
    Note right of NCCL: 环形通信与计算并行执行
    NCCL-->>NCCL: Bucket 0 同步完成
    Bucket->>NCCL: 异步触发 AllReduce (Bucket 1)
    NCCL-->>NCCL: 全部 Bucket 同步完毕
```
<p class="caption" align="center"><em>图 4-2：PyTorch DDP 梯度分桶与异步 AllReduce 计算通信重叠流水</em></p>
"""
    },
    "lesson05.qmd": {
        "title": "第 5 课：ZeRO 与 FSDP",
        "fig1": """
::: {.img-card}
<img src="assets/zero3_comm_flow.jpeg" width="90%" alt="ZeRO-3 (FSDP) 每训练步通信与显存状态图解" />
<p class="caption">图 5-1：ZeRO-3 (FSDP) 每训练步通信与显存状态图解</p>
:::
""",
        "fig2": """
::: {.img-card}
<img src="assets/zero3_comm_flow.svg" width="90%" alt="ZeRO-3 分块状态与通信算子流转全景" />
<p class="caption">图 5-2：ZeRO-3 阶段 1（前向）、阶段 2（反向）、阶段 3（优化器更新）分块状态与通信算子流转全景</p>
:::
"""
    },
    "lesson06.qmd": {
        "title": "第 6 课：张量并行 TP",
        "fig1": """
```mermaid
flowchart LR
    subgraph MegatronMLP["Megatron-LM MLP 列行切分架构"]
        direction TB
        X["输入 X"] --> C0["GPU 0: 列切分 W1_0"]
        X --> C1["GPU 1: 列切分 W1_1"]
        C0 --> Act0["GeLU"]
        C1 --> Act1["GeLU"]
        Act0 --> R0["GPU 0: 行切分 W2_0"]
        Act1 --> R1["GPU 1: 行切分 W2_1"]
        R0 & R1 --> AR["AllReduce (Sum)"]
        AR --> Y["最终输出 Y"]
    end
```
<p class="caption" align="center"><em>图 6-1：Megatron-LM MLP 模块列分片与行分片矩阵乘法拓扑</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant Input as 输入张量 X
    participant Column as Column Linear (QKV Projection)
    participant Core as Multi-Head Attention Compute
    participant Row as Row Linear (Out Projection)
    participant Comm as 通信原语 (AllReduce)
    
    Input->>Column: 无通信广播输入 X
    Column->>Core: 各卡计算本地注意力头 (Heads / TP)
    Core->>Row: 局部注意力上下文乘本地行分片
    Row->>Comm: 触发一次 AllReduce 同步各卡部分和
    Comm-->>Input: 输出完整特征向量进入下层
```
<p class="caption" align="center"><em>图 6-2：张量并行 Attention 模块一次前向传播数据流与通信点</em></p>
"""
    },
    "lesson07.qmd": {
        "title": "第 7 课：序列并行 SP",
        "fig1": """
```mermaid
flowchart TD
    subgraph Comparison["TP vs TP+SP 显存分布对比"]
        subgraph StandardTP["传统 Tensor Parallelism"]
            TP_Act["LayerNorm/Dropout 激活值: 每卡存储完整序列 (b, s, h)"]
            TP_Attn["Attention/MLP: 权重分片，内部激活部分分片"]
        end
        subgraph SP["Sequence Parallelism (Megatron-SP)"]
            SP_Act["LayerNorm/Dropout 沿序列切分: 每卡仅存 (b, s/TP, h)"]
            SP_Comm["在 Attention 入口 AllGather，出口 ReduceScatter"]
        end
    end
```
<p class="caption" align="center"><em>图 7-1：序列并行（SP）切分非张量并行区激活值显存分布对比</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    LN["LayerNorm (序列切分 s/TP)"] -->|AllGather 通信| Attn["Attention (全序列 s 进入 Column Linear)"]
    Attn --> OutProj["Output Projection 行分片"]
    OutProj -->|ReduceScatter 通信| Dropout["Dropout (恢复序列切分 s/TP)"]
```
<p class="caption" align="center"><em>图 7-2：Megatron-SP 巧妙利用 Ring 通信原语消除冗余通信的完整闭环</em></p>
"""
    },
    "lesson08.qmd": {
        "title": "第 8 课：上下文并行 CP",
        "fig1": """
```mermaid
flowchart TD
    subgraph CPArchitecture["Context Parallelism (Ring Attention) 数据切分"]
        Seq["超长序列 (如 128k ~ 1M Tokens)"] --> B0["Card 0: Q0, K0, V0 (0~32k)"]
        Seq --> B1["Card 1: Q1, K1, V1 (32~64k)"]
        Seq --> B2["Card 2: Q2, K2, V2 (64~96k)"]
        Seq --> B3["Card 3: Q3, K3, V3 (96~128k)"]
    end
```
<p class="caption" align="center"><em>图 8-1：上下文并行按序列切块并分配至独立 GPU 拓扑</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant Q as Local Query Block (常驻本地)
    participant KV as Local Key/Value Block
    participant P2P as 异步 P2P 环形发送 (Ring Send/Recv)
    participant Acc as FlashAttention Online Accumulator
    
    Q->>Acc: 与本地 KV0 计算局部注意力
    KV->>P2P: 异步把 KV 发送给下邻居，接收上邻居 KV
    P2P-->>Acc: 用接收到的 KV1 计算注意力，并通过 Online Softmax 配平更新
    Note over Q,Acc: 轮转 N-1 次后完成全序列因果注意力且显存恒定
```
<p class="caption" align="center"><em>图 8-2：Ring-Attention 环形 KV 轮转与双缓冲通信计算重叠流程</em></p>
"""
    },
    "lesson09.qmd": {
        "title": "第 9 课：流水线并行 PP",
        "fig1": """
::: {.img-card}
![](assets/figs/train_pp_bubble.png){width="85%"}
<p class="caption">图 9-1：1F1B 流水线并行空闲气泡（Bubble）占比与 Microbatch 关系曲线（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    subgraph Schedule["1F1B (One-Forward-One-Backward) 调度时序"]
        Warmup["预热阶段: 连续执行 Forward 填满流水线深度 (F1, F2, F3...)"]
        Steady["稳态阶段: 每执行一个 Forward 紧跟一个 Backward (1F1B)，显存恒定"]
        Cooldown["清空阶段: 执行剩余 Backward 释放全部临时激活"]
        Warmup --> Steady --> Cooldown
    end
```
<p class="caption" align="center"><em>图 9-2：1F1B 调度器稳态执行与微批次（Micro-Batch）显存收敛机制</em></p>
"""
    },
    "lesson10.qmd": {
        "title": "第 10 课：专家并行 EP",
        "fig1": """
```mermaid
flowchart TD
    Tokens["输入 Token 序列"] --> Router["Top-K 门控路由网络 (Softmax / Sigmoid)"]
    Router --> E0["专家 0 (Rank 0)"]
    Router --> E1["专家 1 (Rank 1)"]
    Router --> E2["专家 2 (Rank 2)"]
    Router --> E3["专家 3 (Rank 3)"]
```
<p class="caption" align="center"><em>图 10-1：MoE 门控路由网络与跨 GPU 专家切分架构</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant T as 本地 Tokens
    participant R as Gating Router
    participant A2A1 as All-to-All 调度通信
    participant Exp as 跨卡 Expert FFN
    participant A2A2 as All-to-All 恢复通信
    
    T->>R: 计算各 Token 对各专家的亲和度分数
    R->>A2A1: 将 Token 打包根据专家所在 Rank 进行 All-to-All 分发
    A2A1->>Exp: 各卡专家并行计算前向 FFN
    Exp->>A2A2: 结果通过反向 All-to-All 汇聚送回原始输入 Rank
    A2A2-->>T: 按门控权重加权求和，输出混合结果
```
<p class="caption" align="center"><em>图 10-2：MoE 专家并行双向 All-to-All 通信与计算全流程</em></p>
"""
    },
    "lesson11.qmd": {
        "title": "第 11 课：5D 并行",
        "fig1": """
```mermaid
flowchart TD
    Cluster["超大规模集群计算拓扑"] --> TP["TP (机内 NVLink 域, TP=4~8)"]
    Cluster --> SP["SP / CP (长上下文序列维度拆分)"]
    Cluster --> PP["PP (跨机层级切分, PP=4~16)"]
    Cluster --> EP["EP (MoE 稀疏专家层间路由)"]
    Cluster --> DP["DP / ZeRO-3 (全局数据副本分片)"]
    TP & SP & PP & EP & DP ==> Matrix["5D 并行网格: [DP, PP, TP, SP, EP]"]
```
<p class="caption" align="center"><em>图 11-1：5D 并行维度分解与物理通信层次映射架构</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Start["开始集群训练配置搜索"] --> C1{"单卡是否装得下?"}
    C1 -->|是| PureDP["纯 DP / ZeRO-1 (通信效率最高)"]
    C1 -->|否| C2{"单机 8 卡装得下?"}
    C2 -->|是| ZeRO3["ZeRO-3 / FSDP (无跨机模型切分)"]
    C2 -->|否| C3["机内启用 TP=4/8 + 跨机启用 PP=4/8 + 外层叠加 DP/EP"]
    C3 --> Check["验证每层通信带宽瓶颈与计算通信重叠率"]
```
<p class="caption" align="center"><em>图 11-2：工业界大模型并行训练方案选型与切分决策树</em></p>
"""
    },
    "lesson12.qmd": {
        "title": "第 12 课：配置搜索与性能诊断",
        "fig1": """
```mermaid
flowchart LR
    subgraph Roofline["Roofline 性能边界模型"]
        direction TB
        MemBound["Memory Bound (访存受限区): 算术强度 < 阈值，受限于 HBM 带宽"]
        CompBound["Compute Bound (算力受限区): 算术强度 > 阈值，受限于 Tensor Core TFLOPs"]
        Turning["拐点: 硬件峰值算力 / 硬件内存带宽"]
    end
```
<p class="caption" align="center"><em>图 12-1：Roofline 模型算术强度与硬件瓶颈分界图解</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    OOM["训练出现 Out-Of-Memory (OOM) 故障"] --> Check1{"第 1 步还是第 2 步 OOM?"}
    Check1 -->|第 2 步| Reason1["首次 Optimizer Step 产生 Adam FP32 状态或碎片增加"]
    Check1 -->|第 1 步前向| Reason2["Batch Size / Seq Len 过大，激活值峰值击穿显存"]
    Check1 -->|第 1 步反向| Reason3["未开启梯度累积清零或缺少重计算 (Checkpointing)"]
    Reason1 & Reason2 & Reason3 --> Fix["调整 micro-batch / 开启 selective recomputation / 增加并行度"]
```
<p class="caption" align="center"><em>图 12-2：大模型训练常见 OOM 故障根因排查与定位诊断流</em></p>
"""
    },
    "lesson13.qmd": {
        "title": "第 13 课：GPU 内核与 FlashAttention",
        "fig1": """
```mermaid
flowchart TD
    subgraph StandardAttn["传统 Attention: O(N^2) 频繁往返 HBM"]
        QKV["Q, K, V (GMEM)"] --> S["S = Q K^T (写入 HBM)"]
        S --> P["P = Softmax(S) (写入 HBM)"]
        P --> O["O = P V (写回 HBM)"]
    end
    subgraph FlashAttn["FlashAttention: Tile 块驻留 SRAM 融合计算"]
        Q_tile["Q Block"] & K_tile["K Block"] --> SRAM["SRAM 内部 Online Softmax + Matmul"]
        SRAM --> O_tile["直接写回最终 O (节省 10x 访存)"]
    end
```
<p class="caption" align="center"><em>图 13-1：传统注意力大量读写高带宽内存 vs FlashAttention 片上瓦片融合对比</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Outer["外层循环: 遍历 K, V 分块 (加载至 SRAM)"] --> Inner["内层循环: 遍历 Q 分块 (加载至 SRAM)"]
    Inner --> Kernel["计算局部 S = Q K^T 并执行 Online Softmax 状态更新"]
    Kernel --> Acc["增量累加更新局部输出矩阵 O_block"]
    Acc --> Inner
```
<p class="caption" align="center"><em>图 13-2：FlashAttention 前向分块双层循环流水与增量更新流</em></p>
"""
    },
    "lesson14.qmd": {
        "title": "第 14 课：混合精度与综合面试",
        "fig1": """
```mermaid
flowchart LR
    subgraph Formats["数值精度格式位数对比"]
        FP32["FP32: 1 符号位 + 8 指数位 + 23 尾数位 (4 Bytes)"]
        FP16["FP16: 1 符号位 + 5 指数位 + 10 尾数位 (动态范围小，易下溢)"]
        BF16["BF16: 1 符号位 + 8 指数位 + 7 尾数位 (动态范围与 FP32 一致)"]
        FP8["FP8 E4M3 / E5M2: 8 位精度标准 (针对推理与训练前向加速)"]
    end
```
<p class="caption" align="center"><em>图 14-1：现代深度学习主流浮点数值格式位宽与动态范围对比</em></p>
""",
        "fig2": """
```mermaid
sequenceDiagram
    autonumber
    participant Net as 网络前向计算 (BF16)
    participant Loss as 损失计算 (FP32)
    participant Scaler as Loss Scaler (损失缩放器)
    participant Back as 反向求导 (BF16)
    participant Opt as 优化器更新 (FP32 Master Weights)
    
    Net->>Loss: 输出预测并计算标量 Loss
    Loss->>Scaler: 放大 Loss (乘以 scale_factor)
    Scaler->>Back: 反向链式求导，梯度放大避免下溢
    Back->>Scaler: 检查梯度是否出现 Inf / NaN
    alt 无溢出
        Scaler->>Opt: 梯度除以 scale_factor 恢复真实量级，更新参数
        Scaler-->>Scaler: 连续 N 步无溢出则增大 scale_factor
    else 发生溢出
        Scaler->>Opt: 丢弃当前步更新 (Skip Step)
        Scaler-->>Scaler: 减小 scale_factor (除以 2)
    end
```
<p class="caption" align="center"><em>图 14-2：动态损失缩放（Dynamic Loss Scaling）溢出检测与自适应更新时序流</em></p>
"""
    }
}
