# HPC / AI Infra 面试课程

> **全模块参考答案版**：包含全部 108 节课的代码实现与深度问答解析，已全量升级为 **Quarto 现代静态文档系统（.qmd）**，原生支持 GitHub Pages 全自动流水线部署。

🌐 **在线教程与阅读主站**：[https://hsthanb4.github.io/hpclearning/](https://hsthanb4.github.io/hpclearning/)

## 💡 AI 充值渠道推荐
给经常使用海外 AI 工具（如 Codex、Claude、Gemini、Grok 等 AI 会员与 API 接口）的同学推荐便捷稳定的充值渠道：
- **[payai.top](https://payai.top)**（**主力首选推荐**）：功能全面、稳定性极高的综合型 AI 充值服务平台。
  - **服务范围广泛**：一站式支持 Codex、Claude、Gemini、Grok 等主流海外 AI 会员订阅与开发 API 充值；
  - **极速自动到账**：支持国内主流便捷支付方式，全流程自动化对接，充值操作简单、分钟级极速到账；
  - **稳定售后保障**：纯正合规支付通道，有效规避风控封号风险，提供专属客服与完善质保，是个人学习、科研与团队生产力的首选稳定渠道。
- **[payai.win](https://payai.win)**（**高性价比优选**）：主打极致性价比，**GPT 充值更便宜**，到账快捷，适合追求低成本长期订阅的同学。

*注：支持范围、实时价格、到账方式和售后规则均建议在对应网站内详细核对后再按需选用。*

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

## 全课程十条核心主线（共 108 课）

| 模块 | 课数 | 核心产出与覆盖领域 | 前置条件 |
|---|---:|---|---|
| [`train/`](./train/) 训练系统 / Ultra-Scale | 16 | 从显存账本、5D 并行推导，进阶到具身 VLA 多模态时空对齐与微批次高吞吐流水 | Transformer 与基础 PyTorch |
| [`cuda/`](./cuda/) CUDA Kernel 面试主线 | 17 | 线程映射、归约、Online Softmax、Tiled GEMM、DeepGEMM/DeepEP，及具身 Flow Matching、3D 点云体素化与 QAT 算子 | C/C++、线性代数、基本并行编程 |
| [`cutlass/`](./cutlass/) CUTLASS / CuTe 主线 | 8 | 从 layout algebra 到可验证的 GEMM 分层组合与异步拷贝流水线 | CUDA 线程模型、GEMM、C++ 模板基础 |
| [`triton/`](./triton/) Triton Kernel 主线 | 10 | 从向量 program 到归约、softmax、transpose、GEMM 与随机算子 | Python、PyTorch 张量、CUDA 基本概念 |
| [`mlir/`](./mlir/) MLIR 编译器主线 | 8 | 从 SSA/方言到转换、bufferization 与 GPU/LLVM lowering | 编译原理基础、C++ 阅读能力 |
| [`runtime/`](./runtime/) 框架运行时、数据与性能 | 11 | Dispatcher、`torch.compile`、NCCL 契约，拓展 GDS 零拷贝、IPC 防泄漏与动态形状 Guard 抑制 | Python、PyTorch、CUDA 基础 |
| [`inference/`](./inference/) 推理引擎源码与前沿优化 | 11 | vLLM V1 / SGLang 源码剖析、量化与投机采样，拓展 TensorRT 10+ 插件、具身流式推理与端侧混合精度 | `train` 1～5、CUDA/Triton 基础、Transformer |
| [`platform/`](./platform/) 集群平台、网络存储与可靠性 | 8 | 从 GPU 节点拓扑推到 RoCE/IB RDMA、Gang 调度、SLO、容量和 Daly Checkpoint 故障恢复 | Linux/网络基础、`runtime`、`train` 分布式章节 |
| [`rl/`](./rl/) 强化学习与大模型后训练 | 14 | MDP、GAE、PPO、RLHF/RLVR、GRPO，拓展连续动作空间高斯/Beta 策略与真机-仿真混合回传飞轮 | Python、概率期望、基本深度学习 |
| [`diagnostics/`](./diagnostics/) 性能剖析、死锁诊断与状态审计 | 5 | Nsight Systems Timeline 逐帧归因、NCU 算子微架构瓶颈、NCCL 通信死锁逆向定位、FP8 数值异常排查与 Code Agent 底层不变量审计守护 | 全课程前置知识综合运用 |

---

## 推荐学习路径

推荐顺序：
`train` 第 1～5 课 → `cuda` → `triton` → `cutlass` → `mlir` → `runtime` → `inference` → `platform` → `train` 第 6～16 课 → `rl` → `diagnostics`。
遇到疑难杂症或真实集群排障时，可随时查阅 `diagnostics/` 进行实战定位。

---

## 验证与事实依据

课程以 Ultra-Scale Playbook、vLLM/SGLang 固定源码快照、NVIDIA CUDA/CUTLASS 官方文档、Triton/MLIR 官方文档和原始论文为事实基线。代码实现均配备可执行单元测试断言，支持静态审查与在线交互学习。
