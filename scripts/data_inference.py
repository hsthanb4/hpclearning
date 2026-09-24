# -*- coding: utf-8 -*-
"""
Inference Engine & Frontier Optimizations Track (8 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

INFERENCE_LESSONS = {
    "lesson01.qmd": {
        "title": "第 1 课：vLLM V1 源码地图：AsyncLLM、EngineCore 与 Worker",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph Frontend["API Server / Frontend"]
        AsyncLLM["AsyncLLM 异步服务接口"]
        ReqQueue["输入请求队列 (Input Queue)"]
    end
    subgraph Core["EngineCore (单事件循环单线程)"]
        Scheduler["Unified Scheduler (统一调度器)"]
        KVManager["KVCacheManager (显存块分配)"]
    end
    subgraph Backend["分布式 Worker 集群 (WorkerPool)"]
        W0["ModelRunner (GPU 0 - TP Rank 0)"]
        W1["ModelRunner (GPU 1 - TP Rank 1)"]
    end
    AsyncLLM -->|ZMQ / IPC 无锁通道| Core
    Scheduler & KVManager -->|发送执行指令| Backend
```
<p class="caption" align="center"><em>图 1-1：vLLM V1 三层解耦核心架构 (Frontend → EngineCore → Workers)</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Req["HTTP 请求到达"] --> Tokenize["分词器编码 Tokens"]
    Tokenize --> PutReq["压入 EngineCore 请求队列"]
    PutReq --> Step["EngineCore::step() 统一调度决策"]
    Step --> Forward["GPU ModelRunner 并行前向推理 (TP AllReduce)"]
    Forward --> Detokenize["采样器输出 Token 解码并流式回传 SSE"]
```
<p class="caption" align="center"><em>图 1-2：vLLM V1 端到端请求流转与异构协同执行流水线</em></p>
"""
    },
    "lesson02.qmd": {
        "title": "第 2 课：vLLM V1 统一调度器、KVCacheManager 与 Prefix Cache",
        "fig1": """
::: {.img-card}
![](assets/figs/kvcache_memory_growth.png){width="85%"}
<p class="caption">图 2-1：Llama-3 8B 不同并发度与上下文长度下 KV Cache 显存暴涨曲线（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart TD
    ScheduleStart["进入 step() 调度周期"] --> CheckWaiting["检查等待队列 (Waiting Queue)"]
    CheckWaiting --> MatchPrefix["查询 Prefix Cache 命中已有块"]
    MatchPrefix --> Budget{"Token 预算与物理 Block 是否充足以容纳新请求?"}
    Budget -->|充足| Promote["晋升至运行队列 (Running Queue) 触发 Prefill"]
    Budget -->|不足| PreemptCheck{"Running 队列能否满足 Decode 扩张?"}
    PreemptCheck -->|受阻| Preempt["抢占低优先级请求 (Preempt / Recompute)"]
    PreemptCheck -->|充足| Exec["构造当前批次元数据并下发 GPU"]
```
<p class="caption" align="center"><em>图 2-2：vLLM V1 统一调度器动态分发与抢占决策流程</em></p>
"""
    },
    "lesson03.qmd": {
        "title": "第 3 课：SGLang SRT 源码地图：Scheduler、ScheduleBatch 与 ModelRunner",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph TokenizerLayer["TokenizerManager (多进程)"]
        Tok["并行 Tokenizer 编解码"]
    end
    subgraph SchedulerLayer["SRT Scheduler"]
        Batch["ScheduleBatch 动态合批容器"]
        Radix["Radix Tree 索引空间"]
    end
    subgraph ModelLayer["ModelRunner (TP / CUDA Graph)"]
        Graph["CUDA Graph 捕获执行图"]
        Kernels["FlashInfer / PagedAttention 算子"]
    end
    Tok --> SchedulerLayer --> ModelLayer
```
<p class="caption" align="center"><em>图 3-1：SGLang SRT 运行时分层交互与 ScheduleBatch 容器拓扑</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Req["接收批次请求"] --> RadixMatch["Radix Tree 快速最长前缀检索"]
    RadixMatch --> FormBatch["构造混合批次 (Prefill + Decode ScheduleBatch)"]
    FormBatch --> GraphExec{"命中捕获尺寸的 CUDA Graph?"}
    GraphExec -->|是| FastRun["重放 CUDA Graph (超低调度开销)"]
    GraphExec -->|否| EagerRun["回退 Eager 动态形状前向执行"]
    FastRun & EagerRun --> Sample["多项式采样与树节点更新"]
```
<p class="caption" align="center"><em>图 3-2：SGLang SRT 混合批处理与 CUDA Graph 捕获分发时序流程</em></p>
"""
    },
    "lesson04.qmd": {
        "title": "第 4 课：SGLang RadixAttention、内存池与 Overlap Scheduling",
        "fig1": """
```{mermaid}
flowchart LR
    Root["根节点: []"] --> S1["[Hello, World] (Len=2, Ref=3)"]
    S1 --> S2["[How are you?] (Len=4, Ref=1)"]
    S1 --> S3["[Explain CUDA] (Len=3, Ref=2)"]
    subgraph LRU["LRU 淘汰双向链表"]
        Head["最久未访问节点 (Ref=0)"] <--> Tail["最新访问节点"]
    end
    S2 -.->|Ref降为0| Head
```
<p class="caption" align="center"><em>图 4-1：SGLang RadixAttention 基数树与 LRU 引用淘汰拓扑机制</em></p>
""",
        "fig2": """
```{mermaid}
flowchart TD
    subgraph CPU["CPU 调度线程 (Thread 1)"]
        Prep["Batch N+1: Radix 匹配与显存块分配"] --> Prepare["准备输入张量与元数据"]
    end
    subgraph GPU["GPU 计算流 (Stream 1)"]
        ExecN["Batch N: 模型前向计算与 Attention 计算"]
    end
    Prep -->|"重叠并行 (Overlap)"| ExecN
    Prepare -->|流水衔接| ExecNext["Batch N+1: 立即进入 GPU 计算"]
```
<p class="caption" align="center"><em>图 4-2：SGLang CPU 调度准备与 GPU 执行重叠掩盖 (Overlap) 工作流</em></p>
"""
    },
    "lesson05.qmd": {
        "title": "第 5 课：稀疏推理：权重、MoE、Sparse Attention 与 HiSparse",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph SparsityTech["大模型推理稀疏化四维矩阵"]
        Weight["2:4 结构化稀疏 / 细粒度权重剪枝"]
        MoE["混合专家稀疏激活 (如 8 选 2 专家路由)"]
        Attn["块稀疏 Attention (Block-Sparse / Top-K)"]
        HiSparse["分层动态稀疏 (HiSparse KV/FFN 联合压缩)"]
    end
```
<p class="caption" align="center"><em>图 5-1：现代大模型端到端稀疏推理优化维度概念架构</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    TokenIn["Token 激活向量"] --> Router["门控路由网络 Top-K 评分"]
    Router --> Scatter["Token 重排与专家分组 (Permute/Scatter)"]
    Scatter --> GroupGEMM["分专家并行 Grouped GEMM 计算"]
    GroupGEMM --> Gather["加权聚合与逆重排 (Unpermute/Gather)"]
    Gather --> Output["输出融合后激活向量"]
```
<p class="caption" align="center"><em>图 5-2：MoE 动态路由、专家分组计算与跨卡打散聚合流程</em></p>
"""
    },
    "lesson06.qmd": {
        "title": "第 6 课：量化源码走读：INT4、FP8、NVFP4、MXFP4 与 KV Cache",
        "fig1": """
::: {.img-card}
![](assets/figs/quantization_tradeoff.png){width="85%"}
<p class="caption">图 6-1：FP16 vs FP8 vs INT4 显存占用与推理吞吐权衡对比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart TD
    Act["输入激活 X (FP16/BF16)"] --> DynamicQuant["动态在线量化: X_quant = round(X / scale)"]
    Weight["低比特权重 W_quant (离线预量化)"] --> WeightLoad["从显存高效拉取压缩数据 (带宽翻倍)"]
    DynamicQuant & WeightLoad --> TensorCore["Tensor Core 硬件执行低精度点积 (INT4/FP8 GEMM)"]
    TensorCore --> ScaleMul["反量化乘缩放因子: Out = TensorCore * (scale_x * scale_w)"]
    ScaleMul --> Out["高精度写回 (FP16/BF16)"]
```
<p class="caption" align="center"><em>图 6-2：量化 GEMM 在线标定、Tensor Core 计算与反量化流水线</em></p>
"""
    },
    "lesson07.qmd": {
        "title": "第 7 课：投机采样正确性：Draft、Target Verification 与拒绝校正",
        "fig1": """
::: {.img-card}
![](assets/figs/speculative_speedup_model.png){width="85%"}
<p class="caption">图 7-1：投机采样理论加速比与 Token 接受率 α / 草稿步数关系模型（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart LR
    Eval["比对第 i 个候选 Token"] --> ProbCheck{"生成随机数 u ~ U[0,1] < min(1, p_i / q_i) ?"}
    ProbCheck -->|接受| NextToken["接受候选 x_i，继续校验第 i+1 个 Token"]
    ProbCheck -->|拒绝| Resample["拒绝 x_i，从残差分布 max(0, p - q) 重新采样替代词"]
    Resample --> Truncate["截断后续所有投机 Token 并完成本轮提交"]
```
<p class="caption" align="center"><em>图 7-2：严格数学无偏的投机采样拒绝校正与截断决策树流程</em></p>
"""
    },
    "lesson08.qmd": {
        "title": "第 8 课：现代 Speculative Systems：EAGLE-3、MTP、DFlash 与 DSpark",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph EAGLE["EAGLE-3 隐层外推投机"]
        Hidden["前向隐层特征 (Feature Sequence)"] --> Head["自回归轻量预测头"]
    end
    subgraph MTP["DeepSeek MTP (Multi-Token Prediction)"]
        Trunk["主模型主干 (Shared Trunk)"] --> Head1["预测 Token N+1"]
        Trunk --> Head2["预测 Token N+2"]
    end
    EAGLE & MTP --> TreeAttn["树状拓扑注意力掩码验证 (Tree Attention Verification)"]
```
<p class="caption" align="center"><em>图 8-1：现代投机加速体系 (EAGLE 隐层特征 vs MTP 多头直出) 拓扑结构</em></p>
""",
        "fig2": """
```{mermaid}
flowchart TD
    DraftTree["构建投机 Token 候选树 (Tree Mask)"] --> TargetForward["Target 模型执行一次 Tree Attention 并行验证"]
    TargetForward --> SelectPath["沿最佳拓扑路径选择最长接受前缀"]
    SelectPath --> Rollback["回滚未采纳分支的 KV Cache 指针"]
    Rollback --> Commit["提交被验证分支 Token，进入下一循环"]
```
<p class="caption" align="center"><em>图 8-2：基于树形拓扑的投机验证、KV 状态回滚与批量采纳流水线</em></p>
"""
    }
}
