# -*- coding: utf-8 -*-
"""
Reinforcement Learning & Post-Training Track (12 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

RL_LESSONS = {
    "lesson01_mdp_return.qmd": {
        "title": "第 1 课：MDP、回报与 Bellman 递推",
        "fig1": """
```mermaid
flowchart TD
    subgraph MDP["马尔可夫决策过程 (MDP) 状态转移与回报"]
        S["当前状态 s"] -->|执行动作 a ~ π| A["动作 a"]
        A -->|环境转移动力学 P| S_prime["下一状态 s'"]
        A -->|奖励函数 R| R["即时奖励 r = R(s, a)"]
        S_prime -->|折现累加 γ| G["未来回报 G_t = r + γ V(s')"]
    end
```
<p class="caption" align="center"><em>图 1-1：MDP 状态转移五元组与 Bellman 期望方程分解拓扑</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    T["终止时刻 T: G_T = 0"] --> T1["倒数第一步: G_{T-1} = R_{T-1} + γ G_T"]
    T1 --> T2["倒数第二步: G_{T-2} = R_{T-2} + γ G_{T-1}"]
    T2 --> T0["时刻 0: G_0 = 累积折现回报"]
```
<p class="caption" align="center"><em>图 1-2：有限时域轨迹回报逆序动态规划递推流程</em></p>
"""
    },
    "lesson02_mc_td.qmd": {
        "title": "第 2 课：Monte Carlo、TD 与偏差—方差",
        "fig1": """
```mermaid
flowchart TD
    subgraph MC["蒙特卡洛 (MC) 备份"]
        S_mc["状态 S_t"] --> A_mc["完整轨迹采样"] --> End_mc["终止状态 S_T (真实累计 G_t)"]
    end
    subgraph TD["时间差分 (TD(0)) 备份"]
        S_td["状态 S_t"] --> A_td["一步转移"] --> S_next["引导自举 (Bootstrapping) R + γ V(S_{t+1})"]
    end
```
<p class="caption" align="center"><em>图 2-1：蒙特卡洛全轨迹采样与时间差分单步自举机制对比</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    State["输入当前状态 S_t"] --> Pred["价值网络预测 V(S_t)"]
    State --> EnvStep["环境单步交互得到 R_{t+1}, S_{t+1}"]
    EnvStep --> Target["计算 TD 目标: y = R_{t+1} + γ V(S_{t+1})"]
    Pred & Target --> Loss["TD 误差: δ = y - V(S_t)"]
    Loss --> Grad["梯度反传更新网络权重: θ ← θ + α δ ∇V"]
```
<p class="caption" align="center"><em>图 2-2：TD(0) 单步时序差分更新与闭环学习流程</em></p>
"""
    },
    "lesson03_policy_gradient.qmd": {
        "title": "第 3 课：策略梯度与基线",
        "fig1": """
```mermaid
flowchart LR
    State["输入观测 s"] --> PolicyNet["策略网络 π_θ(a|s)"]
    PolicyNet --> Sample["采样动作 a ~ π_θ"]
    Sample --> Env["环境反馈回报 G_t"]
    State --> BaselineNet["价值基线网络 V_ϕ(s)"]
    Env & BaselineNet --> Adv["优势权重: A_t = G_t - V_ϕ(s) (大幅缩减方差)"]
    Adv & PolicyNet --> Grad["∇J(θ) = E[∇log π_θ(a|s) * A_t]"]
```
<p class="caption" align="center"><em>图 3-1：策略梯度定理与状态基线 (Baseline) 减方差数据流</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    Init["初始化策略网络 π_θ 与价值基线 V_ϕ"] --> Rollout["收集多条交互轨迹 Episode 集合"]
    Rollout --> ComputeReturn["逆序计算各步累积回报 G_t"]
    ComputeReturn --> ComputeAdv["计算中心化基线差值 A_t = G_t - V_ϕ(s_t)"]
    ComputeAdv --> UpdatePolicy["执行策略梯度上升更新 θ"]
    ComputeAdv --> UpdateValue["以 MSE 损失拟合基线更新 ϕ"]
```
<p class="caption" align="center"><em>图 3-2：REINFORCE 带基线策略梯度的完整迭代循环流程</em></p>
"""
    },
    "lesson04_gae.qmd": {
        "title": "第 4 课：GAE 与优势估计",
        "fig1": """
::: {.img-card}
![](assets/figs/rl_bias_variance_tradeoff.png){width="85%"}
<p class="caption">图 4-1：广义优势估计 GAE(λ) 偏差-方差权衡与最优点曲线（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    Last["最后一步: A_T^{GAE} = δ_T"] --> Step["递归计算: A_t = δ_t + γ λ (1 - done_t) A_{t+1}"]
    Step --> Norm["全批次标准化: A_t = (A_t - mean) / (std + 1e-8)"]
    Norm --> Out["输出高质量优势信号注入策略损失更新"]
```
<p class="caption" align="center"><em>图 4-2：GAE 逆序递归高效计算与批量标准化流水线</em></p>
"""
    },
    "lesson05_ppo.qmd": {
        "title": "第 5 课：PPO：比率裁剪与训练循环",
        "fig1": """
::: {.img-card}
![](assets/figs/ppo_surrogate_objective.png){width="85%"}
<p class="caption">图 5-1：PPO 悲观裁剪目标函数曲线与截断边界（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart TD
    ActorRollout["旧策略 π_old 采样收集轨迹缓冲 (Buffer)"] --> EvalGAE["计算价值标靶与 GAE 标准化优势"]
    EvalGAE --> EpochLoop["多轮 Epoch 小批次 Mini-batch 迭代"]
    EpochLoop --> ComputeLoss["计算 L_CLIP + c1*L_VF + c2*Entropy"]
    ComputeLoss --> Optimizer["AdamW 更新网络权重 θ"]
    Optimizer --> CheckKL{"KL(π_old || π_θ) > KL_阈值?"}
    CheckKL -->|是| EarlyStop["提前终止当前 Epoch 防止策略崩溃"]
    CheckKL -->|否| Continue["继续下一个 Mini-batch"]
```
<p class="caption" align="center"><em>图 5-2：PPO-Clip 训练内层循环与自适应早停保护时序流程</em></p>
"""
    },
    "lesson06_value_entropy.qmd": {
        "title": "第 6 课：价值损失、熵与多目标优化",
        "fig1": """
```mermaid
flowchart TD
    subgraph MultiObjective["Actor-Critic 联合优化损失函数"]
        L_clip["策略损失: L_CLIP(θ) [最大化期望回报]"]
        L_vf["价值损失: c_1 * L_VF(θ) [最小化均方误差]"]
        L_ent["策略熵奖励: c_2 * S[π_θ] [防止过早收敛/促进探索]"]
    end
    L_clip --> Total["总损失: L = - L_CLIP + c_1 * L_VF - c_2 * S"]
    L_vf --> Total
    L_ent --> Total
```
<p class="caption" align="center"><em>图 6-1：策略、价值与信息熵三合一多任务优化空间</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Early["训练前期: 熵权重大 ➔ 探索未知状态空间"] --> Middle["训练中期: 策略趋于稳定 ➔ 熵系数线性衰减"]
    Middle --> Late["训练后期: 价值误差逼近极小 ➔ 确定性微调策略收敛"]
```
<p class="caption" align="center"><em>图 6-2：强化学习训练过程中策略熵与探索度的动态退火轨迹</em></p>
"""
    },
    "lesson07_dqn.qmd": {
        "title": "第 7 课：DQN：回放、目标网络与过估计",
        "fig1": """
```mermaid
flowchart TD
    subgraph EnvInteraction["环境采样流"]
        Step["(s, a, r, s', done)"] --> Buffer["经验回放缓冲区 (Replay Buffer)"]
    end
    subgraph QNetworks["双网络解耦结构"]
        Buffer -->|均匀随机采样 Batch| TrainQ["在线 Q 网络 Q(s, a; θ)"]
        Buffer --> TargetQ["目标 Q 网络 Q(s', a'; θ^-)"]
        TargetQ -->|计算目标 y| Loss["Bellman 均方差损失"]
        TrainQ -->|预测 Q 值| Loss
    end
    TrainQ -.->|定期同步参数 θ^- = θ| TargetQ
```
<p class="caption" align="center"><em>图 7-1：DQN 经验回放缓冲与目标网络时间解耦架构</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Sample["从 Replay Buffer 采样 (s, a, r, s', d)"] --> DoubleDQN{"是否使用 Double DQN?"}
    DoubleDQN -->|是| Decouple["在线网络选动作 a* = argmax Q(s', .; θ) ➔ 目标网络算价值 Q(s', a*; θ^-)"]
    DoubleDQN -->|否| Standard["目标网络直接计算 max Q(s', .; θ^-) [存在过估计]"]
    Decouple & Standard --> ComputeLoss["计算 Huber/MSE 损失并执行反向传播"]
    ComputeLoss --> Sync{"达到 target_update 步数?"}
    Sync -->|是| SoftSync["更新目标网络权重 θ^-"]
```
<p class="caption" align="center"><em>图 7-2：DQN 与 Double DQN 目标估计及网络参数同步执行流程</em></p>
"""
    },
    "lesson08_sac.qmd": {
        "title": "第 8 课：SAC：最大熵与温度",
        "fig1": """
```mermaid
flowchart TD
    subgraph MaxEntropy["最大熵目标"]
        J["目标: E[∑ r(s,a) + α H(π(·|s))]"]
    end
    subgraph DualQ["Twin Q-Networks (抑制定量偏差)"]
        Q1["Q_1(s, a)"]
        Q2["Q_2(s, a)"]
        MinQ["min(Q_1, Q_2) - α log π(a|s)"]
    end
    J --> MinQ
    Q1 & Q2 --> MinQ
```
<p class="caption" align="center"><em>图 8-1：Soft Actor-Critic (SAC) 最大熵与双 Q 网络架构模型</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    State["状态输入 s"] --> Policy["重参数化采样: a = tanh(μ + σ ⊙ ε)"]
    Policy --> MinQ["双 Q 网络评估当前动作价值"]
    MinQ --> ActorLoss["更新 Actor 策略: ∇_θ [α log π_θ(a|s) - Q_min(s, a)]"]
    ActorLoss --> TempLoss["自适应更新温度 α: ∇_α [-α (log π(a|s) + H_target)]"]
    TempLoss --> Polyak["软更新 (Polyak) 目标网络: θ^- ← τθ + (1-τ)θ^-"]
```
<p class="caption" align="center"><em>图 8-2：SAC 连续动作空间三方网络交替优化与自适应温度调节时序</em></p>
"""
    },
    "lesson09_offline_rl.qmd": {
        "title": "第 9 课：离线 RL 与分布外动作",
        "fig1": """
```mermaid
flowchart LR
    subgraph DataDist["行为策略数据集 D (in-distribution)"]
        D_Points["高频覆盖状态与动作流"]
    end
    subgraph OOD["分布外动作 (Out-of-Distribution)"]
        OOD_Points["未探索动作 ➔ 标准 Q 学习高估虚假峰值"]
    end
    subgraph CQL["保守 Q 学习 (CQL) 正则化"]
        Penalty["对所有动作压低 Q 值 ➔ 仅对数据集动作拉高 Q 值"]
    end
    D_Points & OOD_Points --> CQL
```
<p class="caption" align="center"><em>图 9-1：离线强化学习分布漂移 (Distribution Shift) 与保守 Q 约束原理</em></p>
""",
        "fig2": """
```mermaid
flowchart TD
    LoadData["静态数据集加载 (s, a, r, s') 无在线探索"] --> EstimateQ["Q 网络评估数据集动作价值"]
    EstimateQ --> SampleOOD["在当前状态采样未知潜在动作 (OOD Actions)"]
    SampleOOD --> CQLTerm["计算保守正则项: logsumexp(Q) - E_{D}[Q(s, a)]"]
    CQLTerm --> JointLoss["总损失 = 标准 TD 误差 + α * CQL 惩罚项"]
    JointLoss --> StepOpt["Adam 梯度更新，严格抑制外推误差 (Extrapolation Error)"]
```
<p class="caption" align="center"><em>图 9-2：保守离线 RL (CQL) 算法步进与外推误差控制流程</em></p>
"""
    },
    "lesson10_rlhf.qmd": {
        "title": "第 10 课：RLHF/RLVR：奖励来源、KL 与 token 对齐",
        "fig1": """
```mermaid
flowchart TD
    Prompt["输入 Prompt x"] --> Actor["生成策略模型 π_θ (Active)"]
    Prompt --> Ref["参考基础模型 π_ref (Frozen)"]
    Actor --> Response["生成回复文本 y = (y_1, y_2, ..., y_T)"]
    Response --> RM["奖励模型 r_ϕ(x, y) 或规则验证器 (RLVR)"]
    Actor & Ref --> KL["Token 级 KL 散度: D_KL = log π_θ(y_t) - log π_ref(y_t)"]
    RM & KL --> FinalReward["最终组合标量/密集奖励: R = RM(x, y) - β * D_KL"]
```
<p class="caption" align="center"><em>图 10-1：大模型对齐 RLHF 四模型拓扑与 Token 级 KL 惩罚锚定</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    PromptBatch["采样 Prompt 批次"] --> VLLM["高性能推理引擎生成候选文本"]
    VLLM --> RMScore["奖励模型打分 / 编译器验证 (RLVR)"]
    RMScore --> KLPenalty["计算对参考模型的 KL 漂移惩罚"]
    KLPenalty --> GAEAdv["沿 Token 序列递归生成 GAE 优势信号"]
    GAEAdv --> PPOUpdate["分布式框架 (Megatron/DeepSpeed) 反向传播更新权重"]
```
<p class="caption" align="center"><em>图 10-2：RLHF / RLVR 后训练全链路端到端数据流水线</em></p>
"""
    },
    "lesson11_grpo.qmd": {
        "title": "第 11 课：GRPO 与组内相对优势",
        "fig1": """
::: {.img-card}
![](assets/figs/grpo_reward_distribution.png){width="85%"}
<p class="caption">图 11-1：GRPO 无 Critic 组内相对优势打分分布与标准化（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```mermaid
flowchart LR
    BatchQ["获取一批查询 Questions"] --> Rollout["每个 Query 采样 G 份完整回答"]
    Rollout --> Eval["答案规则匹配 / 单元测试打分 (0/1 或浮点)"]
    Eval --> Norm["计算各 Prompt 组内部的 Z-score 优势"]
    Norm --> Loss["组装 GRPO 截断损失与参考模型 KL 约束"]
    Loss --> Grad["跨卡通信与模型参数更新"]
```
<p class="caption" align="center"><em>图 11-2：GRPO 高吞吐后训练执行与组间优势分配流程</em></p>
"""
    },
    "lesson12_rl_system.qmd": {
        "title": "第 12 课：大规模 RL 系统：rollout、训练与评测",
        "fig1": """
```mermaid
flowchart TD
    subgraph InferenceCluster["Rollout 生成集群 (vLLM / SGLang)"]
        W1["Worker 1 (Prompt 吞吐)"]
        W2["Worker 2 (KV Cache 优化)"]
    end
    subgraph StorageQueue["解耦中转层"]
        Queue["高吞吐经验队列 (Shared Memory / Redis / Ray Object Store)"]
    end
    subgraph TrainingCluster["训练更新集群 (Megatron / ZeRO-3)"]
        L1["Learner GPU 0"]
        L2["Learner GPU 1"]
    end
    W1 & W2 -->|推送采样轨迹| Queue
    Queue -->|拉取 Batch 经验| L1 & L2
    L1 -.->|异步参数广播 / 权重热加载| W1 & W2
```
<p class="caption" align="center"><em>图 12-1：工业级大模型强化学习系统 Rollout 与 Learner 解耦架构</em></p>
""",
        "fig2": """
```mermaid
flowchart LR
    Gen["Rollout 引擎并发生成轨迹"] --> Put["压入异步 FIFO 轨迹队列"]
    Put --> CheckStaleness{"轨迹延迟 Staleness ≤ 阈值?"}
    CheckStaleness -->|有效| Train["Learner 取出执行并行梯度计算"]
    CheckStaleness -->|过期| Drop["丢弃过期轨迹，防止策略梯度发散"]
    Train --> WeightSync["NCCL 广播最新权重至 Rollout Worker 内存"]
```
<p class="caption" align="center"><em>图 12-2：异步 RL 系统经验管道吞吐与策略落后度 (Staleness) 控制流程</em></p>
"""
    }
}
