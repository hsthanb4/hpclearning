# -*- coding: utf-8 -*-
"""
PyTorch Runtime, Data & Performance Track (8 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

RUNTIME_LESSONS = {
    "lesson01.qmd": {
        "title": "第 1 课：PyTorch 执行栈、Dispatcher 与 Autograd",
        "fig1": """
```mermaid
flowchart TD
    subgraph Frontend["Python API 接口层"]
        PyOp["torch.matmul(a, b) / torch.add(x, y)"]
    end
    subgraph Dispatcher["c10::Dispatcher 多键动态路由"]
        Keys["Dispatch Key: Autograd | CPU | CUDA | Tracer | Batched"]
    end
    subgraph Kernels["底层 C++ / CUDA 实现"]
        GradNode["Autograd::AddBackward0 (构造反向图)"]
        ATenCUDA["at::native::cuda_kernel() (GPU 执行)"]
    end
    PyOp --> Keys --> GradNode & ATenCUDA
```
<p class="caption" align="center"><em>图 1-1：PyTorch Dispatcher 调度键分发与执行栈拓扑结构</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Loss["标量损失 loss.backward()"] --> Queue["待处理反向节点工作队列 (ReadyQueue)"]
    Queue --> Pop["弹出当前就绪 Node (如 MulBackward0)"]
    Pop --> Compute["执行反向算子计算输入叶子张量梯度"]
    Compute --> Accum["更新或累加 .grad 属性 (AccumulateGrad)"]
    Accum --> CheckEdges{"检查并递减前驱边依赖计数"}
    CheckEdges -->|就绪| Push["将前驱 Node 入队"]
    CheckEdges -->|未完成| Wait["等待其他路径汇合"]
```
<p class="caption" align="center"><em>图 1-2：Autograd 动态计算图反向拓扑遍历与梯度累加工作流</em></p>
"""
    },
    "lesson02.qmd": {
        "title": "第 2 课：torch.compile：Graph Break、Guard 与动态 Shape",
        "fig1": """
::: {.img-card}
![](assets/figs/pytorch_eager_vs_compile.png){width="85%"}
<p class="caption">图 2-1：torch.compile Inductor 融合算子对比 Eager 原生执行加速比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    Call["调用已编译模型 fn(*args)"] --> CheckGuards{"Guard 快速校验是否全部通过?"}
    CheckGuards -->|命中缓存| FastPath["直接执行已编译高性能 Triton 机器码 (极速)"]
    CheckGuards -->|Guard 失效| Recompile["触发重新捕获与重编译 (Recompilation)"]
    Recompile --> DynamicCheck{"是否出现动态维度漂移 (Dynamic Shapes)?"}
    DynamicCheck -->|是| SymInt["引入符号整数 (SymInt) 生成泛化内核"]
    DynamicCheck -->|否| Specialize["针对当前固定尺寸特化编译"]
```
<p class="caption" align="center"><em>图 2-2：torch.compile Guard 命中判定与自适应特化重编译时序</em></p>
"""
    },
    "lesson03.qmd": {
        "title": "第 3 课：Process Group、Rendezvous 与 Collective 契约",
        "fig1": """
```mermaid
flowchart TD
    subgraph Rendezvous["Rendezvous 发现中心 (TCPStore / etcd)"]
        Store["全局键值协调: Master IP:Port, World Size=8"]
    end
    subgraph Cluster["分布式进程拓扑"]
        R0["Rank 0 (Local 0)"] <--> Store
        R1["Rank 1 (Local 1)"] <--> Store
        R7["Rank 7 (Local 7)"] <--> Store
    end
    subgraph Backends["通信后端 (ProcessGroupNCCL)"]
        Comm["NCCL Communicator (ncclComm_t 唯一定位环与通信拓扑)"]
    end
    Cluster --> Backends
```
<p class="caption" align="center"><em>图 3-1：分布式训练集合通信初始化与 ProcessGroup 物理拓扑映射</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Host["Python 发起 dist.all_reduce(tensor)"] --> Enqueue["向底层 CUDA 通信 Stream 压入 NCCL 内核"]
    Enqueue --> ReturnWork["立即返回异步句柄 (Work Handle) 不阻塞 CPU"]
    ReturnWork --> ComputeStream["计算 Stream 插入 Event 依赖或调用 work.wait()"]
    ComputeStream --> SyncCheck{"GPU 通信是否物理完成?"}
    SyncCheck -->|完成| Next["安全使用 AllReduce 汇总后张量结果"]
    SyncCheck -->|等待| GPUWait["GPU 硬件执行跨卡数据交换与规约"]
```
<p class="caption" align="center"><em>图 3-2：异步集合通信下发、流同步契约与句柄等待流程</em></p>
"""
    },
    "lesson04.qmd": {
        "title": "第 4 课：Collective 成本模型与拓扑选择",
        "fig1": """
::: {.img-card}
![](assets/figs/ring_vs_tree_latency.png){width="85%"}
<p class="caption">图 4-1：Ring 与 Tree 集合通信算法理论时延模型与报文尺度分界（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    Msg["待规约报文尺寸 S 与集群卡数 P"] --> Threshold{"S 是否大于环/树分界阈值?"}
    Threshold -->|S 较小 (小包)| SelectTree["选择 Tree / CollNet 算法 (极小化延迟 α)"]
    Threshold -->|S 较大 (大张量)| SelectRing["选择 Ring 环形切分流水 (极大化总线带宽利用率 β)"]
    SelectTree & SelectRing --> PipeCheck{"跨机网络是否支持 NVLink + RoCE 异构层次?"}
    PipeCheck -->|支持| TwoLevel["启用分层规约: 节点内 NVLink 规约 ➔ 节点间跨机网络 ➔ 节点内广播"]
```
<p class="caption" align="center"><em>图 4-2：通信报文尺度判定与层次化集合通信算法路由决策流</em></p>
"""
    },
    "lesson05.qmd": {
        "title": "第 5 课：数据分片、Shuffle 与精确 Resume",
        "fig1": """
```mermaid
flowchart TD
    subgraph Dataset["全量训练数据集 (N 条记录)"]
        Raw["Samples [0, 1, 2, ..., N-1]"]
    end
    subgraph Sampler["DistributedSampler 确定性分发"]
        Seed["随机种子 + Epoch 序号"] --> ShuffleIndices["确定性伪随机置换序列"]
        ShuffleIndices --> Shard0["Rank 0 数据分片: [idx0, idx4, ...]"]
        ShuffleIndices --> Shard1["Rank 1 数据分片: [idx1, idx5, ...]"]
    end
    Dataset --> Sampler
```
<p class="caption" align="center"><em>图 5-1：分布式数据分片采样与可复现 Shuffle 逻辑映射模型</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Crash["训练异常中断 / 节点抢占故障"] --> LoadCkpt["加载恢复 Checkpoint 元数据"]
    LoadCkpt --> RestoreState["恢复 Epoch 计数器、随机种子与已消费 Batch 偏移量"]
    RestoreState --> FastForward["Sampler 快速跳过当前分片已完成样本 (O(1) 索引计算)"]
    FastForward --> ResumePipe["无损无缝对齐数据加载管道，避免重复消费或样本遗漏"]
```
<p class="caption" align="center"><em>图 5-2：断点续训 (Resume) 精确状态恢复与跳步定位工作流</em></p>
"""
    },
    "lesson06.qmd": {
        "title": "第 6 课：存储→CPU→Pinned Memory→GPU 流水",
        "fig1": """
```mermaid
flowchart LR
    Storage["远端对象存储 / NVMe SSD"] -->|Read/IO| PageCache["操作系统 Page Cache (CPU)"]
    PageCache -->|Decode/Process| PageableRAM["常规 CPU 内存 (可换页分页内存)"]
    PageableRAM -->|Fast Copy| PinnedMem["锁页内存 (Pinned / Page-Locked Memory)"]
    PinnedMem -->|PCIe DMA 极速拷贝| VRAM["GPU 显存 (HBM / GDDR)"]
```
<p class="caption" align="center"><em>图 6-1：数据从外部介质经锁页内存直达 GPU 的四层物理搬运通道</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    subgraph WorkerThreads["DataLoader Background Workers (CPU)"]
        W1["读取下一批原始样本"] --> W2["解码图像/分词张量"]
    end
    subgraph PinnedQueue["Pinned Memory 预取缓冲队列"]
        Q["固定大小 Prefetch 队列"]
    end
    subgraph GPUSync["GPU 异步流水线"]
        DMA["cudaMemcpyAsync(..., stream=copy_stream)"]
        Compute["GPU 前向与反向计算 (compute_stream)"]
    end
    W2 -->|非阻塞压入| Q
    Q -->|DMA 搬运| DMA
    DMA -->|CUDA Event 同步| Compute
```
<p class="caption" align="center"><em>图 6-2：CPU 数据预处理与 GPU 计算三级异步重叠流水线</em></p>
"""
    },
    "lesson07.qmd": {
        "title": "第 7 课：Profiler、Timeline 与 CUDA Allocator",
        "fig1": """
```mermaid
flowchart TD
    subgraph CachingAllocator["PyTorch Caching Allocator 双池架构"]
        Large["大块内存池 (Large Pool: >= 1MB) ➔ 容纳激活与大权重"]
        Small["小块内存池 (Small Pool: < 1MB) ➔ 容纳标量与小临时变量"]
    end
    subgraph Segments["物理显存段 (Segments)"]
        S1["Segment 1 (20MB) ➔ 细分为多个 Allocated/Free Blocks"]
        S2["Segment 2 (50MB) ➔ 细分为多个 Allocated/Free Blocks"]
    end
    CachingAllocator --> Segments
```
<p class="caption" align="center"><em>图 7-1：PyTorch 缓存分配器双池结构与显存切分碎片控制模型</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    StartTrace["启动 torch.profiler 采集"] --> CPUHook["CPU 端拦截 Python Op 分发并打标 Correlation ID"]
    CPUHook --> CUPTI["CUPTI 驱动层捕获真实 GPU Kernel Launch 与执行时延"]
    CUPTI --> MatchEvents["依据 Correlation ID 关联 CPU 算子与 GPU 内核时序"]
    MatchEvents --> ExportJSON["导出 Chrome Trace / Perfetto 格式瀑布流 Timeline"]
```
<p class="caption" align="center"><em>图 7-2：端到端 Profiler 追踪捕获、跨设备事件关联与时间线呈现流程</em></p>
"""
    },
    "lesson08.qmd": {
        "title": "第 8 课：可复现性、回归定位与性能排障",
        "fig1": """
```mermaid
flowchart TD
    subgraph JitterSources["训练性能抖动与吞吐下降核心诱因"]
        Straggler["掉队卡 (Straggler): 降频 / ECC 双位错误重试"]
        NCCLWait["集合通信悬停 (NCCL Barrier Stall): 负载不均"]
        GC["Python GC 停顿与 CPU 调度抖动"]
        IOContention["分布式存储 IO 拥塞导致 GPU 饥饿"]
    end
```
<p class="caption" align="center"><em>图 8-1：分布式集群性能回归与抖动诱因排查矩阵</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Alert["监控报警: 训练 TFLOPS / Step Time 出现异常回归"] --> Step1["抓取全节点 Profiler Timeline 查看 GPU 空闲气泡"]
    Step1 --> CheckBubble{"主要气泡出现在 IO 还是 通信?"}
    CheckBubble -->|IO 等待| FixIO["调优 DataLoader worker 数、开启 pin_memory 或本地缓存"]
    CheckBubble -->|NCCL 等待| CheckRank{"各卡 Step Time 是否严重不均?"}
    CheckRank -->|是| FindStraggler["定位单卡硬件故障 / 动态序列过长木桶短板"]
    CheckRank -->|否| OptComm["排查网络拓扑配对 / 开启通信重叠或梯度分桶"]
```
<p class="caption" align="center"><em>图 8-2：系统级性能回归二分分段排查与根因定位决策流</em></p>
"""
    }
}
