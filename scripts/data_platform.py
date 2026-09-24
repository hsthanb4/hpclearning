# -*- coding: utf-8 -*-
"""
Cluster Platform, Networking, Storage & Reliability Track (8 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

PLATFORM_LESSONS = {
    "lesson01.qmd": {
        "title": "第 1 课：GPU 节点拓扑：NUMA、PCIe、NVLink 与 NIC 亲和性",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph NUMA0["NUMA Node 0 (CPU 0)"]
        RAM0["Local RAM 0"]
        GPU0["GPU 0"] <--> PCIe0["PCIe Switch 0"]
        GPU1["GPU 1"] <--> PCIe0
        NIC0["RDMA NIC 0"] <--> PCIe0
    end
    subgraph NUMA1["NUMA Node 1 (CPU 1)"]
        RAM1["Local RAM 1"]
        GPU2["GPU 2"] <--> PCIe1["PCIe Switch 1"]
        GPU3["GPU 3"] <--> PCIe1
        NIC1["RDMA NIC 1"] <--> PCIe1
    end
    NUMA0 <==>|"UPI / QPI 跨节点互联 (高时延低带宽)"| NUMA1
    GPU0 <==>|"NVSwitch / 全互联 NVLink (超高带宽)"| GPU2
```
<p class="caption" align="center"><em>图 1-1：现代 8 卡 GPU 节点拓扑、NVLink 高速网与 NUMA/NIC 亲和性架构</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Task["调度器绑定 Pod 进程"] --> CheckLoc["检测目标 GPU 物理插槽位置"]
    CheckLoc --> PinCPU["使用 numactl 绑定对应的同侧 NUMA CPU 核心"]
    PinCPU --> PinNIC["绑定挂载在同侧 PCIe Switch 的近端 RDMA NIC"]
    PinNIC --> ZeroUPI["消除跨 NUMA UPI 内存穿透，获得最低端到端延迟与满血带宽"]
```
<p class="caption" align="center"><em>图 1-2：硬件亲和性探测与跨 NUMA/PCIe 访问惩罚规避策略流程</em></p>
"""
    },
    "lesson02.qmd": {
        "title": "第 2 课：InfiniBand、RoCE、RDMA 与 GPUDirect 数据路径",
        "fig1": """
::: {.img-card}
![](assets/figs/gpudirect_vs_host_tcp.png){width="85%"}
<p class="caption">图 2-1：GPUDirect RDMA 与传统 Host TCP/IP 网络吞吐与带宽饱和对比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart TD
    Setup["初始化 InfiniBand 设备与上下文"] --> RegMR["注册内存区域 (ibv_reg_mr 锁定物理页)"]
    RegMR --> CreateQP["创建发送/接收队列对 (Queue Pair: QP)"]
    CreateQP --> Exchange["带外通信 (TCP) 交换 QP 元数据并进入 RTS 状态"]
    Exchange --> PostSend["下发异步传输请求 (ibv_post_send / RDMA Write)"]
    PostSend --> PollCQ["网卡硬件自主搬运并轮询完成队列 (ibv_poll_cq)"]
```
<p class="caption" align="center"><em>图 2-2：RDMA 队列对 (QP) 建立、内存注册与硬件自驱动传输流水线</em></p>
"""
    },
    "lesson03.qmd": {
        "title": "第 3 课：存储层级、数据集与 Checkpoint 带宽",
        "fig1": """
```{mermaid}
flowchart TD
    L1["GPU 显存 (HBM3e: 3~8 TB/s)"] --> L2["Host 内存 (DDR5: 200~400 GB/s)"]
    L2 --> L3["节点本地 NVMe SSD (PCIe Gen5: 10~50 GB/s)"]
    L3 --> L4["共享并行分布式存储 (Lustre / GPFS / JuiceFS: 1~10 GB/s)"]
    L4 --> L5["冷数据归档对象存储 (S3 / OSS: 0.1~1 GB/s)"]
```
<p class="caption" align="center"><em>图 3-1：大模型集群五级存储金字塔层次与带宽特性</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Trigger["触发 Checkpoint 存档"] --> DumpMem["步骤 1: 内存级快照 (GPU -> CPU RAM, ~1秒)"]
    DumpMem --> ResumeTrain["立即恢复训练正反向计算，训练零气泡阻塞"]
    DumpMem --> AsyncDisk["步骤 2: 后台后台线程异步将 RAM 刷写至本地 NVMe"]
    AsyncDisk --> AsyncRemote["步骤 3: 异步流式上传至中央共享对象存储"]
```
<p class="caption" align="center"><em>图 3-2：非阻塞异步多级 Checkpoint 保存流水线与持久化工作流</em></p>
"""
    },
    "lesson04.qmd": {
        "title": "第 4 课：容器、驱动、Device Plugin 与升级兼容性",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph HostOS["宿主机环境 (Host OS)"]
        Kernel["Linux 内核"]
        Driver["NVIDIA 驱动 (nvidia.ko)"]
        Toolkit["nvidia-container-toolkit"]
    end
    subgraph K8s["Kubernetes 集群调度"]
        Kubelet["Kubelet 节点守护进程"]
        DevPlugin["NVIDIA K8s Device Plugin"]
    end
    subgraph Container["应用容器 (Pod Container)"]
        CUDA["CUDA Runtime (libcudart)"]
        PyTorch["PyTorch / AI 训练推理框架"]
    end
    DevPlugin -->|向 Kubelet 上报 nvidia.com/gpu 资源| Kubelet
    Toolkit -->|挂载驱动字符设备 /dev/nvidia*| Container
```
<p class="caption" align="center"><em>图 4-1：NVIDIA 容器运行时生态与 K8s Device Plugin 分层交互拓扑</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Start["Device Plugin 启动"] --> Query["调用 NVML 查询节点可用物理 GPU 数量"]
    Query --> Report["通过 gRPC ListAndWatch 向 Kubelet 注册 gpu 资源池"]
    Report --> Schedule["K8s 调度器分配 Pod 到当前节点"]
    Schedule --> Allocate["Kubelet 调用 DevicePlugin.Allocate() 挑选具体 GPU ID"]
    Allocate --> Hook["容器运行时 prestart hook 注入设备节点并启动容器"]
```
<p class="caption" align="center"><em>图 4-2：Kubernetes GPU 设备发现、配额上报与容器挂载生命周期</em></p>
"""
    },
    "lesson05.qmd": {
        "title": "第 5 课：Gang Scheduling 与拓扑感知放置",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph NonGang["传统调度器 (可能引发死锁)"]
        JobA_Part["任务 A: 申请到 4 卡 (等待另外 4 卡)"]
        JobB_Part["任务 B: 申请到 4 卡 (等待另外 4 卡)"]
        Deadlock["资源互相锁死，均无法启动"]
        JobA_Part -.-> Deadlock
        JobB_Part -.-> Deadlock
    end
    subgraph Gang["Gang 调度器 (All-or-Nothing)"]
        Pool["集群空闲卡池: 4 卡"]
        Queue["任务队列: 任务 A 需 8 卡"]
        Wait["任务 A 保持整体排队，不占用任何资源，彻底避免死锁"]
        Pool & Queue --> Wait
    end
```
<p class="caption" align="center"><em>图 5-1：分布式训练 All-or-Nothing Gang 调度防死锁核心机制</em></p>
""",
        "fig2": """
```{mermaid}
flowchart TD
    Job["新提交分布式作业 (请求 32 卡)"] --> CheckLevel1{"是否有单机柜满足全量卡数?"}
    CheckLevel1 -->|是| PlaceRack["机柜内放置: 极小化跨 ToR 核心交换机流量 (最优)"]
    CheckLevel1 -->|否| CheckLevel2{"是否有同属于同一汇聚交换机的相邻机柜?"}
    CheckLevel2 -->|是| PlaceAgg["同 Spine 汇聚层放置: 保证次优 RDMA 带宽"]
    CheckLevel2 -->|否| Reject["排队等待拓扑资源空出，拒绝散布放置"]
```
<p class="caption" align="center"><em>图 5-2：集群拓扑感知调度放置 (Topology-Aware Scheduling) 决策树</em></p>
"""
    },
    "lesson06.qmd": {
        "title": "第 6 课：容量、配额、公平与 Backfill",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph QuotaPool["集群总资源配额池 (1000 GPUs)"]
        TeamA["部门 A 保证配额 (Guaranteed: 400 GPUs)"]
        TeamB["部门 B 保证配额 (Guaranteed: 400 GPUs)"]
        Shared["共享空闲与突发池 (Oversubscription: 200 GPUs)"]
    end
    subgraph DRF["主导资源公平调度 (Dominant Resource Fairness)"]
        Dominant["根据 CPU、GPU、内存中占比最高的主导份额实现均衡分配"]
    end
    QuotaPool --> DRF
```
<p class="caption" align="center"><em>图 6-1：多租户配额保障、弹性突发超售与 DRF 资源公平模型</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    LargeJob["高优先级大作业等待 64 卡 (预计 30 分钟后到位)"] --> LockTime["调度器计算预留时间窗口 (Reservation Window)"]
    LockTime --> ScanSmall["扫描等待队列中的碎片小作业 (如 4 卡、运行 15 分钟)"]
    ScanSmall --> BackfillCheck{"小作业是否能在预留时间前结束?"}
    BackfillCheck -->|是| RunNow["立即回填调度 (Backfill) 启动小作业，极大化集群利用率"]
    BackfillCheck -->|否| WaitSmall["小作业保持等待，确保绝不推迟大作业启动时刻"]
```
<p class="caption" align="center"><em>图 6-2：回填调度 (Backfill Scheduling) 与保留时间窗口填充工作流</em></p>
"""
    },
    "lesson07.qmd": {
        "title": "第 7 课：Metrics、Logs、Traces 与 SLO Burn Rate",
        "fig1": """
```{mermaid}
flowchart LR
    subgraph Telemetry["集群可观测性三支柱"]
        M["Metrics (GPU 利用率, 温度, ECC, NCCL 吞吐)"]
        L["Logs (内核 dmesg, PyTorch 栈日志, K8s 事件)"]
        T["Traces (分布式跨节点 RPC 与 NCCL 通信耗时)"]
    end
    Telemetry --> BurnRate["SLO 错误预算消耗率 (Burn Rate = 当前故障率 / 允许故障率)"]
```
<p class="caption" align="center"><em>图 7-1：集群立体化监控度量与 SLO 错误预算 (Burn Rate) 监控模型</em></p>
""",
        "fig2": """
```{mermaid}
flowchart TD
    MetricStream["时序指标持续采集 (Prometheus + DCGM-Exporter)"] --> EvalWindow["多窗口联合评估: 1小时消耗率 > 14.4x 或 6小时 > 6x"]
    EvalWindow --> Spike{"是否发生急剧突发严重故障?"}
    Spike -->|"是 (高 Burn Rate)"| P1Alert["触发 P1 紧急告警，直接呼叫值班并启动自动下线隔离"]
    Spike -->|"否 (慢速漂移)"| P3Ticket["创建 P3 低优先级工单排查慢掉队节点"]
    P1Alert --> HotMigration["触发作业热迁移与断点恢复自愈"]
```
<p class="caption" align="center"><em>图 7-2：多窗口 SLO 消耗率告警分级判定与自动化应急响应流程</em></p>
"""
    },
    "lesson08.qmd": {
        "title": "第 8 课：故障恢复、Checkpoint 间隔与多租户事故设计",
        "fig1": """
::: {.img-card}
![](assets/figs/daly_optimal_checkpoint.png){width="85%"}
<p class="caption">图 8-1：万卡集群 MTBF 故障率与 Daly 最优 Checkpoint 周期曲线模型（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart LR
    NodeFault["某计算节点发生 Xid 崩溃或通信断联"] --> Heartbeat["控制面探测器毫秒级发现心跳丢失"]
    Heartbeat --> Cordon["自动隔离污染坏节点 (Cordon & Taint)"]
    Cordon --> HotSpare["从冷备池动态调拨同规格健康节点加入拓扑"]
    HotSpare --> Reload["从最近持久化 Checkpoint 载入权重并重建 NCCL 通信环"]
    Reload --> Resume["作业恢复正常训练，全过程自动化自愈无需人工干预"]
```
<p class="caption" align="center"><em>图 8-2：万卡分布式集群节点崩溃自动容隔离、热替换与自愈恢复全流程</em></p>
"""
    }
}
