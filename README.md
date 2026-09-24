# HPC / AI Infra 面试课程

> **全模块参考答案版**：包含全部 88 节课的代码实现与深度问答解析，已全量升级为 **Quarto 现代静态文档系统（.qmd）**，原生支持 GitHub Pages 全自动流水线部署。

🌐 **在线教程与阅读主站**：[https://hsthanb4.github.io/hpclearning/](https://hsthanb4.github.io/hpclearning/)

---

## ⚡ 本地开发与在线部署

本仓库基于 [Quarto](https://quarto.org/) 构建，结合 `freeze: auto` 机制解决了无 GPU 环境下编译渲染 CUDA/Triton 代码的痛点。

### 1. 本地实时预览与全站编译

在本地开发机安装 Quarto 后，可直接在仓库根目录下运行：

```bash
# 本地启动热重载开发服务器（修改 .qmd 实时刷新网页）
quarto preview

# 静态构建全站静态 HTML（产物位于 _site/ 目录）
quarto render

# 任意单课转回 Jupyter Notebook (.ipynb) 进行交互式调试
quarto convert train/lesson01.qmd --output train/lesson01.ipynb
```

### 2. GitHub Pages 全自动部署流水线

已配置 GitHub Actions 自动化工作流（`.github/workflows/publish.yml`）：
- **自动触发**：每次向 `main` 或 `answer` 分支 `git push` 时，云端自动使用 Quarto 将全站编译并发布到独立的 `gh-pages` 分支；
- **免显卡避坑机制（`freeze`）**：`_quarto.yml` 中开启了 `execute: freeze: auto`，云端构建复用冻结缓存，无需配置昂贵的 GPU 云主机即可 0 报错完成发布；
- **GitHub 仓库开启方法**：仓库页面 -> **Settings** -> **Pages** -> 将 Build and deployment 的 Source 设置为 **Deploy from a branch**，分支选择 `gh-pages` / `/ (root)` 即可。

---

## 使用方式与评分规则

1. 按模块内编号学习，一次只做一课。
2. 每课提交唯一代码填空题、运行输出和 Q1～Q3；环境不可用时明确写“静态审查，运行待验证”。
3. 评分为代码 4 分、三个问答各 2 分；达到 8/10 且无关键误解才通过。
4. `main` 永远只保留题目；`answer` 才包含参考答案。不要在第一次尝试前切到答案分支。

---

## 六条核心主线

| 模块 | 课数 | 核心产出 | 前置条件 |
|---|---:|---|---|
| [`train/`](./train/) 训练系统 / Ultra-Scale | 14 | 能做显存/通信账本并设计 5D 并行训练配置 | Transformer 与基础 PyTorch |
| [`cuda/`](./cuda/) CUDA Kernel 面试主线 | 12 | 从线程映射到归约、tiling 与数值稳定算子。 | C/C++、线性代数、基本并行编程 |
| [`cutlass/`](./cutlass/) CUTLASS / CuTe 主线 | 8 | 从 layout algebra 到可验证的 GEMM 分层组合。 | CUDA 线程模型、GEMM、C++ 模板基础 |
| [`triton/`](./triton/) Triton Kernel 主线 | 10 | 从向量 program 到归约、softmax、transpose、GEMM 与随机算子。 | Python、PyTorch 张量、CUDA 基本线程/内存概念 |
| [`mlir/`](./mlir/) MLIR 编译器主线 | 8 | 从 SSA/方言到转换、bufferization 与可调 lowering。 | 编译原理基础、C++ 阅读能力 |
| [`rl/`](./rl/) 强化学习与大模型后训练 | 12 | 从 MDP、REINFORCE/RLOO 到 RLHF/RLVR、GRPO 变体及 rollout 数据契约。 | Python、概率期望、基本深度学习 |

---

## 三条 AI Infra 补充线

这三条线是对六条核心主线的压缩补充，不重复 CUDA kernel、训练并行推导或基础强化学习。每条 8 课，共 24 课。

| 模块 | 课数 | 核心产出 | 前置条件 |
|---|---:|---|---|
| [`inference/`](./inference/) vLLM/SGLang 源码与前沿推理 | 8 | 能走读两套引擎核心源码，并分析稀疏、量化与投机采样的正确性和工程选型。 | `train` 1～5、CUDA/Triton 基础、Transformer |
| [`runtime/`](./runtime/) 框架运行时、数据与性能 | 8 | 能解释 PyTorch 执行/编译/分布式合同，设计数据流水并基于 trace 排障。 | Python、PyTorch、`train` 1～5、CUDA 基础 |
| [`platform/`](./platform/) 集群平台、网络存储与可靠性 | 8 | 能从 GPU 节点拓扑推到调度、SLO、容量和故障恢复。 | Linux/网络基础、`runtime`、`train` 分布式章节 |

- `inference`：vLLM V1 进程架构 → vLLM Scheduler/KV → SGLang SRT → RadixAttention/Overlap → 稀疏推理 → 量化源码 → 投机采样正确性 → EAGLE-3/MTP/DFlash/DSpark。
- `runtime`：Dispatcher/Autograd → `torch.compile` → Process Group → Collective 成本与拓扑 → 数据分片/恢复 → 输入流水 → Profiler/Allocator → 可复现性与回归定位。
- `platform`：节点拓扑 → RDMA/GPUDirect → 存储/Checkpoint → 容器与设备 → Gang/拓扑调度 → 容量/公平/Backfill → 可观测性/SLO → 故障恢复与多租户。

推荐顺序：`train` 第 1～5 课 → `cuda` → `triton` → `cutlass` → `mlir` → `runtime` → `inference` → `platform`；`rl` 可在具备 PyTorch 基础后并行学习，最后回到 `train` 第 6～14 课做综合系统设计。

---

### 强化学习主线的 LLM 阅读路径

第 1 课把 token MDP 与完整回答 bandit 对齐 → 第 3～5 课串起 REINFORCE/RLOO、GAE 与 PPO → 第 9～10 课区分离线 DPO、RLHF/RLVR 和奖励粒度 → 第 11～12 课分析 GRPO 变体、训练/推理分布差异及 Agent 动作 mask。第 2、6～8 课保留通用 RL 基础，不用 LLM 配方替代其他任务的算法。

[Cameron R. Wolfe 的 LLM RL 综述](https://cameronrwolfe.substack.com/p/llm-rl)用于串联阅读：策略梯度是共同骨架；不同方法主要改变学习信号、更新约束与样本权重。课程中的公式与方法边界以各课链接的原论文和官方文档为基线，不把综述中的前沿实验结论当作普遍保证。

---

## 仓库级验证记录（2026-08-12）

- 训练系统：14 课参考实现已在 Python 3.10 / PyTorch 2.6.0 逐课运行通过；Ring AllReduce 模拟额外核对了每卡通信量公式。
- CUDA：12 课参考实现已用 CUDA 12.6、`-arch=sm_80` 编译；第 12 课 ArgMax 已在远程 CUDA 兼容设备运行并与 CPU 参考一致。
- CUTLASS/CuTe：17 个纯 Python 检查已运行；3 个 C++/CuTe 片段已使用 CUTLASS 头文件和 CUDA 12.6 编译。其余流水/epilogue 课程是静态设计练习，不声称做过完整 GEMM benchmark。
- Triton：10 课参考脚本已通过 Python 语法检查。远程设备的 Triton 后端要求 `PPU_SDK/irformatter`，当前环境缺失，GPU JIT/数值验证仍待在标准 NVIDIA Triton 环境完成。
- MLIR：当前本地与远程均无 `mlir-opt/mlir-tblgen`，仅完成文本、类型和 API 静态审查，实际工具链验证待完成。
- 强化学习：12 课纯 Python 参考实现及断言已逐课运行通过；这里验证的是教学算法，不等同于大规模 RLHF 集群 benchmark。
- 推理引擎源码线：8 课参考实现及断言已逐课运行通过；源码固定到 2026-08-12 的 vLLM `8e958902` 与 SGLang `9deb6952`，未把论文峰值或源码存在冒充目标工作负载的 benchmark 结论。
- Runtime/Platform 补充线：16 课纯 Python 参考实现及断言已逐课运行通过；这些验证覆盖教学模型与边界检查，不等同于目标 GPU 集群或网络的生产 benchmark。
