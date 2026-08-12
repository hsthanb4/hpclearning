# HPC / AI Infra 面试课程

题目版：只包含讲解、一个代码填空题和三个待答问题。

## 使用方式

1. 按模块内编号学习，一次只做一课。
2. 每课提交唯一代码填空题、运行输出和 Q1～Q3；环境不可用时明确写“静态审查，运行待验证”。
3. 评分为代码 4 分、三个问答各 2 分；达到 8/10 且无关键误解才通过。
4. `main` 永远只保留题目；`answer` 才包含参考答案。不要在第一次尝试前切到答案分支。

## 六条主线

| 模块 | 课数 | 核心产出 | 前置 |
|---|---:|---|---|
| [`train/`](./train/) 训练系统 / Ultra-Scale | 14 | 能做显存/通信账本并设计 5D 并行训练配置 | Transformer 与基础 PyTorch |
| [`cuda/`](./cuda/) CUDA Kernel 面试主线 | 12 | 从线程映射到归约、tiling 与数值稳定算子。 | C/C++、线性代数、基本并行编程 |
| [`cutlass/`](./cutlass/) CUTLASS / CuTe 主线 | 8 | 从 layout algebra 到可验证的 GEMM 分层组合。 | CUDA 线程模型、GEMM、C++ 模板基础 |
| [`triton/`](./triton/) Triton Kernel 主线 | 10 | 从向量 program 到归约、softmax、transpose、GEMM 与随机算子。 | Python、PyTorch 张量、CUDA 基本线程/内存概念 |
| [`mlir/`](./mlir/) MLIR 编译器主线 | 8 | 从 SSA/方言到转换、bufferization 与可调 lowering。 | 编译原理基础、C++ 阅读能力 |
| [`rl/`](./rl/) 强化学习与大模型后训练 | 12 | 从 MDP 到大规模 RLHF/GRPO 系统，面向算法与系统面试。 | Python、概率期望、基本深度学习 |

推荐顺序：`train` 第 1～5 课 → `cuda` → `triton` → `cutlass` → `mlir`；强化学习可在具备 PyTorch 基础后并行学习，最后回到 `train` 第 6～14 课做综合系统设计。

## 初始进度

- 学习中：`train/lesson01`
- 未开始：其余 63 课
- 待验证：CUDA/CUTLASS/Triton/MLIR 专用工具链；仓库内会记录实际完成的代表性远程 GPU 验证，不把静态检查冒充运行结果。

## 资料原则

课程以 Ultra-Scale Playbook、NVIDIA CUDA/CUTLASS 文档、Triton 官方教程、MLIR 官方文档和原始 RL 论文为事实基线。软件 API 会演进，使用时应以目标环境版本的官方文档为准。

## 仓库级验证记录（2026-08-12）

- 训练系统：14 课参考实现已在 Python 3.10 / PyTorch 2.6.0 逐课运行通过；Ring AllReduce 模拟额外核对了每卡通信量公式。
- CUDA：12 课参考实现已用 CUDA 12.6、`-arch=sm_80` 编译；第 12 课 ArgMax 已在远程 CUDA 兼容设备运行并与 CPU 参考一致。
- CUTLASS/CuTe：17 个纯 Python 检查已运行；3 个 C++/CuTe 片段已使用 CUTLASS 头文件和 CUDA 12.6 编译。其余流水/epilogue 课程是静态设计练习，不声称做过完整 GEMM benchmark。
- Triton：10 课参考脚本已通过 Python 语法检查。远程设备的 Triton 后端要求 `PPU_SDK/irformatter`，当前环境缺失，GPU JIT/数值验证仍待在标准 NVIDIA Triton 环境完成。
- MLIR：当前本地与远程均无 `mlir-opt/mlir-tblgen`，仅完成文本、类型和 API 静态审查，实际工具链验证待完成。
- 强化学习：12 课纯 Python 参考实现及断言已逐课运行通过；这里验证的是教学算法，不等同于大规模 RLHF 集群 benchmark。
