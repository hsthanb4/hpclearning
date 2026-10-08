# -*- coding: utf-8 -*-
"""
Scientific Figure Generator for HPC & AI Infra Core Mathematical Principles.
Generates publication-quality, high-DPI diagrams illustrating key mathematical
and architectural concepts for Train, CUDA, Inference, RL and other tracks.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Canonical Palette from scientific-figure-making
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#E3F2FD",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#4CAF50",
    "green_dark": "#1B5E20",
    "red_1": "#F6CFCB",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "neutral_light": "#F8FAFC",
    "slate": "#475569",
    "slate_light": "#ECEFF1",
    "highlight": "#FFD700",
    "orange": "#E65100",
    "orange_light": "#FFF3E0",
    "purple": "#7B1FA2",
    "purple_light": "#F3E5F5",
}

def set_style():
    plt.rcParams.update({
        "font.family": ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.edgecolor": "#CCCCCC",
        "axes.linewidth": 1.0,
    })

# ----------------------------------------------------------------------
# 1. Train Lesson 07: Sequence Parallelism (Megatron SP vs Ulysses)
# ----------------------------------------------------------------------
def make_fig_train_sp():
    set_style()
    fig = plt.figure(figsize=(14, 7.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "序列并行（Sequence Parallelism）显存消除数学原理与两种主流范式对比", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, r"数学核心: 激活显存从 $O(B \cdot S \cdot d)$ 降低为 $O(B \cdot \frac{S}{\mathrm{TP}} \cdot d)$", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Panel 1: Megatron SP
    rect_left = patches.FancyBboxPatch((3.5, 30), 45, 59, boxstyle="round,pad=0.4,rounding_size=1",
                                      facecolor="#F8FAFC", edgecolor=PALETTE["blue_secondary"], linewidth=1.5)
    ax.add_patch(rect_left)
    ax.text(26, 85.5, "范式 A：Megatron-LM 序列并行 (LayerNorm/Dropout 拆分)", 
            ha="center", fontsize=11.5, fontweight="bold", color=PALETTE["blue_main"])

    ax.text(5.5, 80.5, "• 核心发现：在 TP 中，Attention 和 MLP 内部激活已切分，但 LayerNorm 和 Dropout\n  仍持有全量序列 (B, S, d)，占据了 50% 以上的激活显存冗余！", fontsize=9.2, color="#334155")

    # Megatron dataflow
    ax.add_patch(patches.Rectangle((6, 68), 12, 8, facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.2))
    ax.text(12, 72, "LayerNorm\n" + r"$(B, \frac{S}{\mathrm{TP}}, d)$", ha="center", va="center", fontsize=9, fontweight="bold", color=PALETTE["blue_main"])

    ax.annotate("", xy=(22, 72), xytext=(18.5, 72), arrowprops=dict(arrowstyle="->", color=PALETTE["slate"], lw=2))
    ax.text(20.25, 74.5, "AllGather", ha="center", fontsize=8.5, fontweight="bold", color=PALETTE["orange"])

    ax.add_patch(patches.Rectangle((22.5, 68), 12, 8, facecolor=PALETTE["orange_light"], edgecolor=PALETTE["orange"], linewidth=1.2))
    ax.text(28.5, 72, "Attention (TP)\n" + r"$(B, S, \frac{d}{\mathrm{TP}})$", ha="center", va="center", fontsize=9, fontweight="bold", color=PALETTE["orange"])

    ax.annotate("", xy=(38.5, 72), xytext=(35, 72), arrowprops=dict(arrowstyle="->", color=PALETTE["slate"], lw=2))
    ax.text(36.75, 74.5, "ReduceScatter", ha="center", fontsize=8, fontweight="bold", color=PALETTE["green_dark"])

    ax.add_patch(patches.Rectangle((39, 68), 8, 8, facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.2))
    ax.text(43, 72, "Dropout\n" + r"$(B, \frac{S}{\mathrm{TP}}, d)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["blue_main"])

    txt_megatron = (
        "【代数巧妙闭环】：\n"
        "原 TP 在 Attention 开头无通信、末尾做 AllReduce (Sum)。\n"
        "Megatron SP 将末尾的 AllReduce 拆解为：\n"
        r"  AllReduce = ReduceScatter + AllGather" + "\n"
        "将 ReduceScatter 留在注意力末尾（直接产出切分的序列），在下一个线性层前才做 AllGather！\n"
        "总通信量与原 TP 完全相同（2 倍环规约量），通信次数不变，纯赚 TP 倍显存下降！"
    )
    ax.text(5.5, 63, txt_megatron, fontsize=8.8, color="#1E293B", va="top")

    # Panel 2: DeepSpeed Ulysses
    rect_right = patches.FancyBboxPatch((51.5, 30), 45, 59, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor=PALETTE["purple"], linewidth=1.5)
    ax.add_patch(rect_right)
    ax.text(74, 85.5, "范式 B：DeepSpeed Ulysses 序列并行 (All-to-All 维度转置)", 
            ha="center", fontsize=11.5, fontweight="bold", color=PALETTE["purple"])

    ax.text(53.5, 80.5, "• 核心发现：利用现代 Transformer 头数较多特性，通过 All-to-All 算子在\n  “序列切分维度 S/P” 与 “注意力头切分维度 H/P” 之间自由切换：", fontsize=9.2, color="#334155")

    # Ulysses flow
    ax.add_patch(patches.Rectangle((54, 68), 11, 8, facecolor=PALETTE["purple_light"], edgecolor=PALETTE["purple"], linewidth=1.2))
    ax.text(59.5, 72, "输入 Q, K, V\n" + r"$(\frac{S}{P}, H, d_k)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["purple"])

    ax.annotate("", xy=(69, 72), xytext=(65.5, 72), arrowprops=dict(arrowstyle="->", color=PALETTE["slate"], lw=2))
    ax.text(67.25, 74.5, "All-to-All", ha="center", fontsize=8.5, fontweight="bold", color=PALETTE["purple"])

    ax.add_patch(patches.Rectangle((69.5, 68), 12, 8, facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.2))
    ax.text(75.5, 72, "Local Attention\n" + r"$(S, \frac{H}{P}, d_k)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["green_dark"])

    ax.annotate("", xy=(85.5, 72), xytext=(82, 72), arrowprops=dict(arrowstyle="->", color=PALETTE["slate"], lw=2))
    ax.text(83.75, 74.5, "All-to-All", ha="center", fontsize=8.5, fontweight="bold", color=PALETTE["purple"])

    ax.add_patch(patches.Rectangle((86, 68), 9, 8, facecolor=PALETTE["purple_light"], edgecolor=PALETTE["purple"], linewidth=1.2))
    ax.text(90.5, 72, "输出投影\n" + r"$(\frac{S}{P}, H, d_k)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["purple"])

    txt_ulysses = (
        "【适用场景与通信特征】：\n"
        "• 计算核心可直接调用原生 FlashAttention-2/3，完全保留单卡极致内核算力！\n"
        "• 2 次 All-to-All 通信，每张卡通信量与进程数 P 线性对齐；\n"
        "• 约束条件：头数 H 必须能被 P 整除（在 GQA 下受限于 KV 头数）；\n"
        "• 适合支持数十万级甚至百万级超长上下文（Context Window）的超长文本扩展。"
    )
    ax.text(53.5, 63, txt_ulysses, fontsize=8.8, color="#1E293B", va="top")

    # Bottom summary box
    rect_bot = patches.FancyBboxPatch((3.5, 4), 93, 23, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#FFFBEB", edgecolor="#D97706", linewidth=1.2)
    ax.add_patch(rect_bot)
    ax.text(6, 23.5, "序列并行（SP）数学显存量化对比公式：", fontsize=11, fontweight="bold", color="#B45309")
    eq_summary = (
        r"• 标准 TP 未开启 SP：每个 Transformer 层的显存占用为: " + "\n"
        r"  M_act = B * S * d * ( 10 + 24/TP + 5 * a*S / (d * TP) ) 字节" + "\n"
        r"• 开启 SP 之后：所有非注意力算子（LayerNorm, Bias, Dropout）的激活全部缩减为 1/TP: " + "\n"
        r"  M_act_SP = (B * S * d / TP) * ( 10 + 24 + 5 * a*S / d )" + "\n"
        "★ 总结：SP 与 TP 深度绑定且完全零额外通信开销，使得 128K+ 长上下文训练成为可能！"
    )
    ax.text(6, 20.5, eq_summary, fontsize=9.2, color="#92400E", va="top")

    p = Path("train/assets/figs/train_sp_megatron_vs_ulysses.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 2. CUDA Lesson 07: Online Softmax (FlashAttention Core)
# ----------------------------------------------------------------------
def make_fig_cuda_online_softmax():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "Online Softmax 增量更新与基准重标定数学原理 (FlashAttention 核心)", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学目标: 在无需全量 GMEM 往返的前提下，将分块局部 Softmax 精确递推合并为全局真解", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Left: Block 1
    ax.add_patch(patches.Rectangle((5, 54), 25, 33, facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(17.5, 83.5, "分块 1：历史局部状态 (Tile 1)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    ax.text(7, 78, r"输入向量分块: $x^{(1)} = [x_1, \dots, x_K]$", fontsize=9.5, color="#1E293B")
    ax.text(7, 73, "• 局部最大值:\n" + r"  $m_1 = \max(x^{(1)})$", fontsize=9.5, color="#1E293B")
    ax.text(7, 66, "• 局部指数和:\n" + r"  $d_1 = \sum_{j} \exp(x_j^{(1)} - m_1)$", fontsize=9.5, color="#1E293B")
    ax.text(7, 59, "• 局部注意力累加值:\n" + r"  $O_1 = \sum_j P_{ij}^{(1)} V_j^{(1)}$", fontsize=9.5, color="#1E293B")

    # Plus
    ax.text(34, 70, r"$\bigoplus$", ha="center", va="center", fontsize=24, color=PALETTE["orange"], fontweight="bold")
    ax.text(34, 64, "增量合并", ha="center", va="center", fontsize=9, color=PALETTE["orange"], fontweight="bold")

    # Center: Block 2
    ax.add_patch(patches.Rectangle((38, 54), 25, 33, facecolor=PALETTE["orange_light"], edgecolor=PALETTE["orange"], linewidth=1.5))
    ax.text(50.5, 83.5, "分块 2：新到达数据分块 (Tile 2)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["orange"])
    ax.text(40, 78, r"输入向量分块: $x^{(2)} = [x_{K+1}, \dots, x_{2K}]$", fontsize=9.5, color="#1E293B")
    ax.text(40, 73, "• 新分块最大值:\n" + r"  $m_2 = \max(x^{(2)})$", fontsize=9.5, color="#1E293B")
    ax.text(40, 66, "• 新分块指数和:\n" + r"  $d_2 = \sum_{j} \exp(x_j^{(2)} - m_2)$", fontsize=9.5, color="#1E293B")
    ax.text(40, 59, "• 新分块局部注意力:\n" + r"  $\widetilde{O}_2 = \sum_j P_{ij}^{(2)} V_j^{(2)}$", fontsize=9.5, color="#1E293B")

    # Arrow to merged
    ax.annotate("", xy=(70, 70), xytext=(65, 70), arrowprops=dict(arrowstyle="->", color=PALETTE["green_dark"], lw=3))

    # Right: Merged global state
    ax.add_patch(patches.Rectangle((71, 54), 25, 33, facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.8))
    ax.text(83.5, 83.5, "合并后全局真解 (Merged State)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["green_dark"])
    ax.text(73, 78, "1. 全局新最大值:\n" + r"   $m_{\mathrm{new}} = \max(m_1, m_2)$", fontsize=9.5, fontweight="bold", color=PALETTE["green_dark"])
    ax.text(73, 69, "2. 指数配平分母递推:\n" + r"   $d_{\mathrm{new}} = d_1 e^{m_1 - m_{\mathrm{new}}} + d_2 e^{m_2 - m_{\mathrm{new}}}$", fontsize=9.2, fontweight="bold", color=PALETTE["green_dark"])
    ax.text(73, 60, "3. 输出加权状态动态回退重标定:\n" + r"   $O_{\mathrm{new}} = O_1 \cdot \frac{d_1 e^{m_1 - m_{\mathrm{new}}}}{d_{\mathrm{new}}} + \widetilde{O}_2 \cdot \frac{e^{m_2 - m_{\mathrm{new}}}}{d_{\mathrm{new}}}$", fontsize=8.8, fontweight="bold", color=PALETTE["green_dark"])

    # Bottom detailed explanation
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 45, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 45, "数学等价性质与工程代数证明（为什么这能够彻底颠覆经典 Softmax？）：", fontsize=11, fontweight="bold", color="#0F172A")

    proof_text = (
        "1. 经典 Softmax 的访存死结：\n"
        r"   定义 S(x_i) = exp(x_i - m) / \sum exp(x_j - m)。为了保证数值稳定必须先减全局最大值 m = max(x)。" + "\n"
        "   单卡若分块处理，第一块计算时根本不知道全局最大值是多少！因此传统实现必须 Pass 1 扫描求全局 m，Pass 2 扫描求分母 d，Pass 3 归一化写回。\n"
        "   这导致中间巨大的注意力矩阵 S x S 必须完整写回慢速 HBM 显存，产生 O(S^2) 的显存占用与高昂访存开销！\n\n"
        "2. Online Softmax 的代数突破（Milakov & Gimelshein, 2018 / FlashAttention, 2022）：\n"
        r"   核心在于利用指数衰减因子 exp(m_1 - m_new) 实时对历史累积量进行【事后补偿重标定】！" + "\n"
        r"   因为 m_new >= m_1，所以差值 m_1 - m_new <= 0，指数衰减因子永远不会数值溢出！" + "\n"
        "   每当在 GPU SRAM 中读入一个新的数据块，只需用标量乘法将历史维护的 (d_1, O_1) 缩放配平，即可与当前块无缝融合。\n\n"
        "3. 终极系统收益：\n"
        "   • 完全无需在 HBM 中保留哪怕一个字节的中间注意力矩阵！显存开销从 O(S^2) 降低到仅依赖隐藏维度的 O(S)！\n"
        "   • 所有 Softmax 与 GEMM 融合在极高速的片上 SRAM（Shared Memory）内一次完成（Single-Pass Fused Kernel）。"
    )
    ax.text(6, 42, proof_text, fontsize=9.0, color="#334155", va="top")

    p = Path("cuda/assets/figs/cuda_online_softmax_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 3. CUDA Lesson 10: Tiled GEMM Math & Memory Layout
# ----------------------------------------------------------------------
def make_fig_cuda_gemm_tiling():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "CUDA 分块矩阵乘法（Tiled GEMM）分层分块代数推导与数据复用数学模型", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学本质: 利用共享内存（SRAM）分块缓存，将全局显存读取次数从 2MNK 骤降至 2MNK / T", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Matrix A
    ax.add_patch(patches.Rectangle((6, 50), 16, 28, facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(14, 64, "矩阵 $A$\n$(M \\times K)$", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    ax.add_patch(patches.Rectangle((6, 68), 16, 7, facecolor="#90CAF9", edgecolor=PALETTE["blue_main"], linewidth=2.0))
    ax.text(14, 71.5, "Tile $A[m, k]$\n$(T \\times T)$", ha="center", va="center", fontsize=9, fontweight="bold", color="#0D47A1")

    ax.text(26, 64, r"$\times$", ha="center", va="center", fontsize=22, color=PALETTE["slate"], fontweight="bold")

    # Matrix B
    ax.add_patch(patches.Rectangle((30, 50), 28, 28, facecolor=PALETTE["orange_light"], edgecolor=PALETTE["orange"], linewidth=1.5))
    ax.text(50, 58, "矩阵 $B$\n$(K \\times N)$", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["orange"])
    ax.add_patch(patches.Rectangle((37, 50), 7, 28, facecolor="#FFCC80", edgecolor=PALETTE["orange"], linewidth=2.0))
    ax.text(40.5, 71.5, "Tile $B[k, n]$\n$(T \\times T)$", ha="center", va="center", fontsize=9, fontweight="bold", color="#E65100")

    ax.text(62, 64, r"$=$", ha="center", va="center", fontsize=22, color=PALETTE["slate"], fontweight="bold")

    # Matrix C
    ax.add_patch(patches.Rectangle((66, 50), 28, 28, facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.5))
    ax.text(80, 64, "矩阵 $C$\n$(M \\times N)$", ha="center", va="center", fontsize=11, fontweight="bold", color=PALETTE["green_dark"])
    ax.add_patch(patches.Rectangle((73, 68), 7, 7, facecolor="#A5D6A7", edgecolor=PALETTE["green_dark"], linewidth=2.0))
    ax.text(76.5, 71.5, "Tile $C[m, n]$\n$(T \\times T)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["green_dark"])

    # Bottom mathematical breakdown
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 42, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 42, "分块矩阵乘法代数恒等式与显存带宽节约比推导：", fontsize=11, fontweight="bold", color="#0F172A")

    proof_gemm = (
        "1. 严格代数分块展开式：\n"
        r"   设将大矩阵 A 沿列、B 沿行按分块步长 T 拆解为 K/T 个连续切片：C = \sum_{k=0}^{K/T - 1} A^{(k)} * B^{(k)}。" + "\n"
        r"   对于输出子块 C[m, n] (T x T)，其每个元素的精确计算公式为：" + "\n"
        r"   C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} B_{k, j} = \sum_{p=0}^{K/T - 1} ( \sum_{t=0}^{T-1} A_{i, p*T+t} * B_{p*T+t, j} )" + "\n"
        "   内部括号内的部分和（T 次浮点乘加）由 Block 内所有线程在寄存器中极速累加，外部循环仅需迭代 K/T 次。\n\n"
        "2. 访存放大倍数（Arithmetic Intensity）与数据复用比：\n"
        "   • 朴素无分块：计算每个点 C_{i,j} 需要读 A 的整行（K 次）和 B 的整列（K 次）。总 GMEM 访存量为 2MNK 次，算术强度极低（仅 ~0.25 FLOP/Byte）；\n"
        "   • 引入 Shared Memory 分块缓存后：每个 Block 内的 T^2 个线程协同加载 T x T 的 A_tile 与 B_tile 进 SRAM；\n"
        "     块内每个元素在计算中被协同复用了 T 次！全局 HBM 读取总量减少至 2MNK / T，带宽需求降低了整整 T 倍（当 T=32 时带宽开销下降 97%）！"
    )
    ax.text(6, 39, proof_gemm, fontsize=9.0, color="#334155", va="top")

    p = Path("cuda/assets/figs/cuda_gemm_tiling_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 4. Train Lesson 04: DDP Ring AllReduce Math & Complexity
# ----------------------------------------------------------------------
def make_fig_train_ddp_ring_allreduce():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "数据并行（DDP）Ring AllReduce 通信拓扑与代数复杂度数学证明", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, r"数学核心: 单卡通信量 $S = 2 \times \frac{P-1}{P} \times M < 2M$，通信总时间与节点数 $P$ 完全解耦", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Panel Left: Phase 1 Scatter-Reduce
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor=PALETTE["blue_secondary"], linewidth=1.5))
    ax.text(26, 85, "阶段 1：Scatter-Reduce 规约分散（分块流水）", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    ax.text(6, 80.5, "• 将大小为 $M$ 的梯度张量均匀切分为 $P$ 个分块 (Chunks)；\n• 环状拓扑（Ring Topology）顺时针同步传递，共执行 $P-1$ 步；\n• 在每一步中，每个 GPU 向下家发送 1 个分块，同时从上家接收并累加：", fontsize=9.0, color="#1E293B")

    # 4 GPUs ring representation
    g_positions = [(12, 60), (36, 60), (36, 52), (12, 52)]
    g_names = ["GPU 0", "GPU 1", "GPU 2", "GPU 3"]
    for (gx, gy), name in zip(g_positions, g_names):
        ax.add_patch(patches.Circle((gx, gy), 3.2, facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.2))
        ax.text(gx, gy, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["blue_main"])

    ax.annotate("", xy=(32, 60), xytext=(16, 60), arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=2))
    ax.annotate("", xy=(36, 56), xytext=(36, 56.5), arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=2))
    ax.annotate("", xy=(16, 52), xytext=(32, 52), arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=2))
    ax.annotate("", xy=(12, 56.5), xytext=(12, 56), arrowprops=dict(arrowstyle="->", color=PALETTE["orange"], lw=2))

    # Panel Right: Phase 2 All-Gather
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor=PALETTE["green_dark"], linewidth=1.5))
    ax.text(74, 85, "阶段 2：All-Gather 全收集广播（真解同步）", ha="center", fontsize=11, fontweight="bold", color=PALETTE["green_dark"])
    ax.text(54, 80.5, "• 此时每张卡持有一个分块的完整全局累加和真解；\n• 同样沿环流动传递真解分块，共执行 $P-1$ 步；\n• 在每一步中，每个 GPU 转发自己收到的已规约分块，覆盖本地局部数据：", fontsize=9.0, color="#1E293B")

    # 4 GPUs ring representation
    g_positions_r = [(60, 60), (84, 60), (84, 52), (60, 52)]
    for (gx, gy), name in zip(g_positions_r, g_names):
        ax.add_patch(patches.Circle((gx, gy), 3.2, facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.2))
        ax.text(gx, gy, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=PALETTE["green_dark"])

    ax.annotate("", xy=(80, 60), xytext=(64, 60), arrowprops=dict(arrowstyle="->", color=PALETTE["green_dark"], lw=2))
    ax.annotate("", xy=(84, 56), xytext=(84, 56.5), arrowprops=dict(arrowstyle="->", color=PALETTE["green_dark"], lw=2))
    ax.annotate("", xy=(64, 52), xytext=(80, 52), arrowprops=dict(arrowstyle="->", color=PALETTE["green_dark"], lw=2))
    ax.annotate("", xy=(60, 56.5), xytext=(60, 56), arrowprops=dict(arrowstyle="->", color=PALETTE["green_dark"], lw=2))

    # Bottom Box: Rigorous Mathematical Derivation
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "严格代数通信量与扩展性证明（为什么 Ring AllReduce 完胜 Parameter Server？）：", fontsize=11, fontweight="bold", color="#0F172A")

    proof_ring = (
        "1. 单卡通信传输量精确推导：\n"
        r"   • 阶段 1 (Scatter-Reduce)：执行 (P - 1) 步，每步传输数据量为 M / P。单卡累计发送量为：S_1 = (P - 1) * (M / P) 字节；" + "\n"
        r"   • 阶段 2 (All-Gather)：执行 (P - 1) 步，每步传输数据量为 M / P。单卡累计发送量为：S_2 = (P - 1) * (M / P) 字节；" + "\n"
        r"   • 单卡总传输量：S_total = S_1 + S_2 = 2 * (P - 1) / P * M < 2M 字节！" + "\n\n"
        "2. 与中心化参数服务器（Parameter Server）对比：\n"
        r"   • 参数服务器（PS 架构）：主节点需接收 (P - 1) 个节点的梯度并分发，主节点通信带宽为 O(P * M)，节点规模增加时主节点立即饱和崩溃；" + "\n"
        r"   • Ring AllReduce 架构：单卡带宽占用恒定受限于 2M，与节点数 P 完全无关！当卡数从 8 卡扩展到 1024 卡时，单卡通信量始终 < 2M！" + "\n\n"
        "3. 通信耗时模型与线性加速比：\n"
        r"   • 传输耗时模型：T_comm = 2 * (P - 1) * \alpha + 2 * \frac{P - 1}{P} * \frac{M}{\beta} （\alpha 为网络延迟，\beta 为网卡双向物理带宽）；" + "\n"
        "   • 当梯度尺寸 M 足够大（如 25MB 以上的 Bucket）时，延迟项可忽略，通信完全受限于网卡带宽，实现近乎理想的超线性弱扩展比！"
    )
    ax.text(6, 38, proof_ring, fontsize=9.0, color="#334155", va="top")

    p = Path("train/assets/figs/train_ddp_ring_allreduce_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 5. Inference Lesson 01: Prefill vs Decode Duality Math Model
# ----------------------------------------------------------------------
def make_fig_inference_prefill_vs_decode():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "大模型推理双阶段计算/访存二象性数学模型（Prefill 计算受限 vs Decode 显存墙受限）", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学本质: Prefill 算术强度高 (AI ~ S/2 FLOP/B) 趋向算力饱和；Decode 算术强度低 (AI ~ 1 FLOP/B) 撞上显存墙", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Left: Prefill
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(26, 85, "阶段 1：首字填充阶段 (Prefill / Prompt 阶段)", ha="center", fontsize=11.5, fontweight="bold", color=PALETTE["blue_main"])
    txt_prefill = (
        "• 算子形态：稠密矩阵乘法 (GEMM)\n"
        r"  输入张量: $X \in \mathbb{R}^{B \cdot S \times d}$，权重 $W \in \mathbb{R}^{d \times d_{\mathrm{out}}}$" + "\n"
        r"• 浮点计算量: $\mathrm{FLOPs} = 2 \cdot (B \cdot S) \cdot d \cdot d_{\mathrm{out}}$" + "\n"
        r"• 显存访存量: $\mathrm{Bytes} = 2 \cdot (d \cdot d_{\mathrm{out}} + B \cdot S \cdot d)$ (FP16)" + "\n"
        r"• 算术强度: $\mathrm{AI} \approx \frac{2 \cdot B \cdot S \cdot d^2}{2 d^2 + 2 B \cdot S \cdot d} \approx B \cdot S$ FLOP/Byte" + "\n"
        "• 硬件工况：当 $B \\cdot S \\geq 256$ 时，算术强度远超硬件拐点 (>150)；\n"
        "  完全处于 Compute-Bound 饱和区，Tensor Cores 利用率高达 60%~80%！"
    )
    ax.text(6, 81, txt_prefill, fontsize=9.2, color="#1E293B", va="top")

    # Right: Decode
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["red_1"], edgecolor=PALETTE["red_strong"], linewidth=1.5))
    ax.text(74, 85, "阶段 2：自回归生成阶段 (Decode / Token 阶段)", ha="center", fontsize=11.5, fontweight="bold", color=PALETTE["red_strong"])
    txt_decode = (
        "• 算子形态：矩阵-向量乘法 (GEMV)\n"
        r"  输入张量: $x \in \mathbb{R}^{B \times 1 \times d}$ (每次仅产生 1 个 Token!)" + "\n"
        r"• 浮点计算量: $\mathrm{FLOPs} = 2 \cdot B \cdot 1 \cdot d \cdot d_{\mathrm{out}}$" + "\n"
        r"• 显存访存量: 必须全量读取整个模型的参数权重 $W$ (数十到数百 GB)！" + "\n"
        r"• 算术强度: $\mathrm{AI}_{\mathrm{decode}} = \frac{2 \cdot B \cdot d^2}{2 d^2} = B \cdot 1.0$ FLOP/Byte" + "\n"
        "• KV Cache 显存墙：还需全量读入累积的历史 $S$ 个 KV Cache 向量：\n"
        r"  $M_{\mathrm{kv}} = 2 \times 2 \times n_{\mathrm{layers}} \times n_{\mathrm{heads}} \times d_k \times S$ 字节！" + "\n"
        "• 硬件工况：极度 Memory-Bound，Tensor Cores 算力利用率通常不足 2%！"
    )
    ax.text(54, 81, txt_decode, fontsize=9.2, color="#1E293B", va="top")

    # Bottom: System optimizations
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "现代推理引擎打破“显存墙”的核心工程数学对策：", fontsize=11, fontweight="bold", color="#0F172A")
    txt_sol = (
        "1. 连续批处理 (Continuous Batching / Orca / vLLM)：\n"
        "   动态合并不同请求的 Decode 步骤，将 GEMV 提升为 Batch GEMM，将有效 Batch Size 扩大至 B=64~128，使算术强度成倍提升，摊薄权重访存成本。\n\n"
        "2. PagedAttention 显存物理块分页：\n"
        "   将连续的 KV Cache 离散化为物理页块 (Blocks)，彻底消除显存内部与外部碎片，将 KV 缓存利用率从 <40% 提升至 >96%，容纳更大 Batch。\n\n"
        "3. 投机采样 (Speculative Decoding / EAGLE)：\n"
        "   利用极小草稿模型 (Draft) 快速猜测 K 个 Token，大模型通过单次前向并行验证 (Verification)，将 K 次高延迟 GEMV 合并为 1 次大 GEMM，突破自回归延迟下界！"
    )
    ax.text(6, 37.5, txt_sol, fontsize=9.0, color="#334155", va="top")

    p = Path("inference/assets/figs/inference_prefill_vs_decode_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 6. RL Lesson 03: Policy Gradient Theorem Math
# ----------------------------------------------------------------------
def make_fig_rl_policy_gradient():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "强化学习策略梯度定理（Policy Gradient Theorem）与对数导数技巧数学推导", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学本质: 巧妙利用对数微分恒等式消除环境转移概率，将未知动力学下的梯度转化为可蒙特卡洛采样的显式期望", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Left: Score function & Log-derivative
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(26, 85, "1. 对数导数技巧 (Log-Derivative / Score Function)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    txt_score = (
        r"• 优化目标：最大化期望累积回报 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$" + "\n"
        r"• 展开为积分形式: $J(\theta) = \int P(\tau; \theta) R(\tau) d\tau$" + "\n"
        r"• 求梯度: $\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau$" + "\n"
        r"• 引入重要代数恒等式: $\nabla f(x) = f(x) \nabla \log f(x)$" + "\n"
        r"  $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$" + "\n"
        r"• 代回积分: $\nabla_\theta J(\theta) = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau$" + "\n"
        r"  $= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]$"
    )
    ax.text(6, 81.5, txt_score, fontsize=9.0, color="#1E293B", va="top")

    # Right: Environment dynamics cancellation
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.5))
    ax.text(74, 85, "2. 环境转移概率神奇相消 (Transition Dynamics Disappears)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["green_dark"])
    txt_cancel = (
        r"• 轨迹完整概率密度展开：" + "\n"
        r"  $P(\tau; \theta) = p(s_0) \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$" + "\n"
        r"• 取对数：$\log P(\tau; \theta) = \log p(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T-1} \log P$" + "\n"
        r"• 对参数 $\theta$ 求梯度：状态转移概率 $P(s_{t+1}|s_t, a_t)$ 与策略参数完全无关！" + "\n"
        r"  $\nabla_\theta \log p(s_0) = 0, \quad \nabla_\theta \log P(s_{t+1}|s_t, a_t) = 0$！" + "\n"
        r"• 最终优雅精简结果：" + "\n"
        r"  $\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)$" + "\n"
        "★ 结论：无需知道黑盒物理环境的动力学方程，即可计算出策略参数的真实梯度！"
    )
    ax.text(54, 81.5, txt_cancel, fontsize=9.0, color="#1E293B", va="top")

    # Bottom: Baseline & Advantage
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "方差缩减定理：基准线（Baseline）不变量与优势函数（Advantage）推导：", fontsize=11, fontweight="bold", color="#0F172A")
    txt_baseline = (
        r"1. 基准线无偏性数学证明：" + "\n"
        r"   引入仅依赖状态的基线 b(s_t)，其梯度的期望恒等于零：" + "\n"
        r"   \mathbb{E}_{a_t \sim \pi} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t) \right] = b(s_t) \sum_{a_t} \pi_\theta(a_t|s_t) \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)} = b(s_t) \nabla_\theta \left( \sum_{a_t} \pi_\theta(a_t|s_t) \right) = b(s_t) \nabla_\theta (1) = 0！" + "\n\n"
        r"2. 广义优势函数 (GAE / Advantage) 的引入：" + "\n"
        r"   减去状态价值基线 V(s_t) 得到优势函数 A(s_t, a_t) = Q(s_t, a_t) - V(s_t)。" + "\n"
        "   • 若动作表现优于平均（A > 0），提升该动作的对数概率；若劣于平均（A < 0），降低其概率；\n"
        "   • 这一数学不变性保证了策略梯度的期望绝对无偏，同时将采样方差骤降 1~2 个数量级，构成了 PPO 与 GRPO 的理论基石！"
    )
    ax.text(6, 37.5, txt_baseline, fontsize=9.0, color="#334155", va="top")

    p = Path("rl/assets/figs/rl_policy_gradient_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 7. CUDA Lesson 13: DeepSeek DeepGEMM FP8 Fine-Grained Math
# ----------------------------------------------------------------------
def make_fig_cuda_deepgemm_fp8():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "DeepSeek DeepGEMM FP8 细粒度分组缩放与片上微架构数学模型", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学本质: 采用 Per-Token (1 x 128) 激活缩放与 Per-Channel (128 x 128) 权重缩放，彻底消除 FP8 动态范围溢出", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Activation scale
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(26, 85, "1. 激活矩阵 A (1 x 128 细粒度分组缩放)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    txt_act = (
        r"• 传统张量级量化 (Tensor-wise Scale) 痛点：" + "\n"
        "  大模型中存在异常离群值 (Outliers)，全张量单一缩放因子会导致小数值欠载下溢 (Underflow)。\n"
        r"• DeepGEMM 方案：沿收缩维度 K 每 128 个元素划为一个分组：" + "\n"
        r"  $A_{\mathrm{fp8}}[i, k] = \mathrm{round}\left( \frac{A[i, k]}{\alpha_i^{(k/128)}} \right)$" + "\n"
        r"  其中 $\alpha_i^{(g)} = \frac{\max_{j \in [0, 127]} |A[i, 128g + j]|}{448.0}$ (FP8 E4M3 最大动态范围)。" + "\n"
        "• 缩放因子开销极低，仅占原激活数据量的 1/128 (不足 1%)！"
    )
    ax.text(6, 81.5, txt_act, fontsize=9.0, color="#1E293B", va="top")

    # Weight scale
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["orange_light"], edgecolor=PALETTE["orange"], linewidth=1.5))
    ax.text(74, 85, "2. 权重矩阵 B (128 x 128 二维瓦片分组缩放)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["orange"])
    txt_wt = (
        r"• 权重离线量化瓦片设计：" + "\n"
        r"  沿输入维度 K 与输出维度 N 双向按 128 x 128 划分二维 Tile；" + "\n"
        r"  $B_{\mathrm{fp8}}[k, j] = \mathrm{round}\left( \frac{B[k, j]}{\beta^{(k/128, j/128)}} \right)$" + "\n"
        r"• 瓦片代数收缩合并：" + "\n"
        r"  计算子块矩阵乘时，由于分组大小刚好与 Tensor Core 瓦片对齐：" + "\n"
        r"  $C[i, j] = \sum_g \left( \alpha_i^{(g)} \cdot \beta^{(g, j/128)} \right) \cdot \sum_{k=0}^{127} A_{\mathrm{fp8}}[i, 128g+k] \cdot B_{\mathrm{fp8}}[128g+k, j]$" + "\n"
        "• 缩放因子的乘法完全移出收缩内积循环，在寄存器中延迟做标量融合！"
    )
    ax.text(54, 81.5, txt_wt, fontsize=9.0, color="#1E293B", va="top")

    # Bottom: TMA & Hopper Specialization
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "硬件级极致加速：TMA 异步张量搬运与 Warp Specialization 架构：", fontsize=11, fontweight="bold", color="#0F172A")
    txt_tma = (
        "1. TMA (Tensor Memory Accelerator) 硬件直通：\n"
        "   无需线程在寄存器中计算全局索引并执行 LDG 指令！TMA 硬件单元直接在后台将全局显存的 2D Tile 数据以异步 DMA 方式搬运至 Shared Memory，\n"
        "   彻底释放 CUDA Core 整数计算单元用于后续的 Epilogue 与反量化标量乘法。\n\n"
        "2. Warp Specialization 生产者-消费者解耦流水：\n"
        "   • Producer Warp (1 个 Warp)：专职维护 TMA 异步描述符与跨迭代预取，驱动硬件多级双缓冲 (Multi-stage Pipeline)；\n"
        "   • Consumer Warp (多 Warp)：专职执行 WGMMA (Warp Group MMA) 指令，直接从 Shared Memory 执行 FP8 矩阵乘加，算力利用率逼近理论峰值！"
    )
    ax.text(6, 37.5, txt_tma, fontsize=9.0, color="#334155", va="top")

    p = Path("cuda/assets/figs/cuda_deepgemm_fp8_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 8. CUDA Lesson 14: DeepSeek DeepEP MoE Dispatch/Combine Math
# ----------------------------------------------------------------------
def make_fig_cuda_deepep_moe():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "DeepSeek DeepEP 专家并行（MoE）低延迟分级通信与非对称调度数学模型", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学本质: 分离机内 NVLink 超高带宽域与机间 RDMA 域，通过非对称 Push/Pull 调度消除 All-to-All 长尾延迟", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Left: Hierarchical Dispatch
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(26, 85, "1. 分层分发通信拓扑 (Hierarchical Dispatch)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    txt_disp = (
        "• 传统扁平 All-to-All 灾难：\n"
        r"  当集群规模扩展到 $N$ 节点、$G$ 卡时，单卡直接与 $N \cdot G$ 个目标建立连接，小包拥塞极其严重。\n"
        "• DeepEP 两级分层通信拓扑：\n"
        r"  (1) 机内 NVLink 聚合：单机 8 卡内部先将发往同一外部目标节点的 Token 聚合；" + "\n"
        r"  (2) 机间 RDMA 单通道批传输：以满载大包 (Large Packets) 穿透 InfiniBand 交换机；" + "\n"
        r"  (3) 目标机 NVLink 二次分发：到达目标机后，本地 GPU 沿 NVLink 极速解包分发给对应 Expert。" + "\n"
        "• 通信有效吞吐提升 3~5 倍，彻底消除跨机小包拥塞！"
    )
    ax.text(6, 81.5, txt_disp, fontsize=9.0, color="#1E293B", va="top")

    # Right: Asymmetric Combine
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.5))
    ax.text(74, 85, "2. 非对称推挽调度 (Asymmetric Push / Pull Dispatch)", ha="center", fontsize=11, fontweight="bold", color=PALETTE["green_dark"])
    txt_comb = (
        "• 为什么 Dispatch 用 Push，Combine 需优化？\n"
        "  - Dispatch 阶段：发送端根据路由门控明确知道每个 Token 的目标专家，采用 Push 模式效率最高；\n"
        "  - Combine 阶段：专家计算完毕后将隐层状态累加回原始 Token，接收端需汇聚来自不同专家的输出。\n"
        "• 低延迟专用 SM 调度器：\n"
        "  - DeepEP 将部分 SM (例如 1~2 个 SM) 专职配置为 RDMA/NVLink 驱动引擎；\n"
        "  - 主计算 SM 专职执行 Expert GEMM，实现通信与计算在微秒级别的无缝重叠 (Zero Bubble Overhead)。"
    )
    ax.text(54, 81.5, txt_comb, fontsize=9.0, color="#1E293B", va="top")

    # Bottom: Latency & Cost Model
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "专家并行（EP）通信复杂度与延迟数学模型：", fontsize=11, fontweight="bold", color="#0F172A")
    txt_model = (
        r"1. 通信数据量建模：" + "\n"
        r"   设序列长度为 S，隐藏维度为 d，Top-K 路由参数为 K_route，专家并行度为 EP。" + "\n"
        r"   每张卡发送的数据量为：V = S * K_route * d * 2 字节（FP16/BF16）或 1 字节（FP8 传输）。" + "\n"
        "   由于 DeepSeek-V3 采用 256 专家且选 8，Top-K 激活的稀疏度极高，单卡传输量大幅受控于分发效率。\n\n"
        "2. 重叠隐蔽条件（Computation-Communication Overlap Condition）：\n"
        r"   只要专家矩阵计算耗时 T_gemm = \frac{2 \cdot N_{\mathrm{tokens}} \cdot d \cdot d_{\mathrm{expert}}}{\mathrm{FLOPS}_{\mathrm{peak}}} > T_{\mathrm{comm}} = \frac{V}{\mathrm{BW}_{\mathrm{network}}}，" + "\n"
        "   整个通信耗时即可被计算完全隐蔽，实现接近零开销的万卡超大规模 MoE 线性扩展！"
    )
    ax.text(6, 37.5, txt_model, fontsize=9.0, color="#334155", va="top")

    p = Path("cuda/assets/figs/cuda_deepep_moe_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 9. Train Lesson 08: Context Parallelism & Ring Attention Math
# ----------------------------------------------------------------------
def make_fig_train_cp_ring_attention():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "上下文并行（Context Parallelism / Ring Attention）环形流水线与因果掩码平衡数学模型", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学核心: 序列切分 S/P 结合 Online Softmax 动态结合律；Zig-Zag 双向折叠消除因果掩码 50% 计算气泡", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Left: Ring Attention Mechanism
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(26, 85, "1. 环形 KV 轮转与 Online Softmax 结合律", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    txt_ring = (
        r"• 切分规则：超长序列 $S$ 沿 Token 维度等分为 $P$ 块，每卡分配 $S/P$；" + "\n"
        r"• $Q$ 块常驻：Rank $i$ 永久持有自身的 Query 瓦片 $Q_i$；" + "\n"
        r"• $K, V$ 环形接力：$K, V$ 瓦片在 GPU 环形链路中循环流动 $P-1$ 次；" + "\n"
        r"• 每次迭代 $k$：Rank $i$ 计算 $S^{(k)} = Q_i (K_{j})^T$ 并局部 Softmax；" + "\n"
        r"• 增量合并状态 $(m, d, O)$：基于 Online Softmax 严格结合律：" + "\n"
        r"  $m_{\mathrm{new}} = \max(m_{\mathrm{prev}}, m_{\mathrm{curr}})$" + "\n"
        r"  $d_{\mathrm{new}} = d_{\mathrm{prev}} e^{m_{\mathrm{prev}}-m_{\mathrm{new}}} + d_{\mathrm{curr}} e^{m_{\mathrm{curr}}-m_{\mathrm{new}}}$" + "\n"
        r"  $O_{\mathrm{new}} = O_{\mathrm{prev}} \frac{d_{\mathrm{prev}} e^{\Delta m_1}}{d_{\mathrm{new}}} + O_{\mathrm{curr}} \frac{d_{\mathrm{curr}} e^{\Delta m_2}}{d_{\mathrm{new}}}$" + "\n"
        "• 遍历完成后，各卡输出精确等于全序列注意力的数学真解！"
    )
    ax.text(6, 81.5, txt_ring, fontsize=8.8, color="#1E293B", va="top")

    # Right: Causal Mask Zig-Zag Balance
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["orange_light"], edgecolor=PALETTE["orange"], linewidth=1.5))
    ax.text(74, 85, "2. 因果掩码下三角气泡与 Zig-Zag 对称平衡", ha="center", fontsize=11, fontweight="bold", color=PALETTE["orange"])
    txt_zigzag = (
        "• 朴素 Ring 在因果掩码下的严重失衡：\n"
        r"  自回归因果掩码下，第 $i$ 块 Query 只能关注历史 $j \leq i$ 块；" + "\n"
        "  Rank 0 仅需计算 1 块，Rank P-1 需计算 P 块，平均气泡高达 50%！\n"
        "• Zig-Zag / Striped Attention 对称折叠解法：\n"
        r"  将序列切分为 $2P$ 块，每张卡分配首尾配对的两个不连续块：" + "\n"
        r"  Rank 0: Chunk [0] 与 Chunk [2P - 1]" + "\n"
        r"  Rank 1: Chunk [1] 与 Chunk [2P - 2]" + "\n"
        r"  Rank $i$: Chunk [$i$] 与 Chunk [$2P - 1 - i$]" + "\n"
        "• 数学对称完美性：\n"
        r"  每张卡分配到的总因果有效注意力块数恒为：$i + 1 + (2P - 1 - i + 1) = 2P + 1$；" + "\n"
        "  各卡计算负载完全严格对称，彻底将因果气泡消除至 0%！"
    )
    ax.text(54, 81.5, txt_zigzag, fontsize=8.8, color="#1E293B", va="top")

    # Bottom: Complexity
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "上下文并行（CP）显存与通信复杂度数学量化：", fontsize=11, fontweight="bold", color="#0F172A")
    txt_cp_math = (
        r"1. 显存复杂度降低 $P$ 倍：" + "\n"
        r"   传统单卡无法承载超长序列（显存占用随序列长度线性爆炸）。开启 CP 后，每张卡激活值恒定降为 $M_{\mathrm{act}} = O(B \cdot \frac{S}{P} \cdot d)$。" + "\n"
        "   这使得即使在 80GB 显卡集群上，也能将上下文长度扩展到 1M (100 万) Tokens 以上！\n\n"
        "2. 通信耗时完全被计算隐蔽的充分条件：\n"
        r"   环形轮转中单步 P2P 通信量为 $V_{\mathrm{step}} = 2 \times \frac{S}{P} \times d \times 2$ 字节（传输 K 和 V 块）。" + "\n"
        r"   单步局部注意力计算量为 $F_{\mathrm{step}} = 4 \times (\frac{S}{P})^2 \times d$ FLOPs。" + "\n"
        r"   当分块长度 $\frac{S}{P} \geq 2048$ 时，局部计算耗时显著超越网络双向传输耗时，实现 100% 理想通信掩盖！"
    )
    ax.text(6, 37.5, txt_cp_math, fontsize=9.0, color="#334155", va="top")

    p = Path("train/assets/figs/train_cp_ring_attention_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# ----------------------------------------------------------------------
# 10. Triton Lesson 09: Triton 2D Matmul PID Swizzle Math
# ----------------------------------------------------------------------
def make_fig_triton_matmul_pid_swizzle():
    set_style()
    fig = plt.figure(figsize=(14, 7.8), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5))

    ax.text(50, 95.5, "Triton 分块矩阵乘法（2D Matmul）分块指针代数与 Program ID 超分组 Swizzling 局部性数学模型", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, "数学本质: 2D Block 广播指针步进；通过 Grouped PID 调度将 L2 Cache 数据复用率提升 2~4 倍", 
            ha="center", va="center", fontsize=11.5, color="#334155")

    # Left: 2D Pointer Arithmetic
    ax.add_patch(patches.FancyBboxPatch((4, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=1.5))
    ax.text(26, 85, "1. Triton 2D 分块指针广播步进代数", ha="center", fontsize=11, fontweight="bold", color=PALETTE["blue_main"])
    txt_ptr = (
        r"• 块内行/列相对偏移生成：" + "\n"
        r"  $\mathrm{offs\_m} = \mathrm{pid\_m} \cdot B_M + \mathrm{arange}(0, B_M)$" + "\n"
        r"  $\mathrm{offs\_n} = \mathrm{pid\_n} \cdot B_N + \mathrm{arange}(0, B_N)$" + "\n"
        r"  $\mathrm{offs\_k} = \mathrm{arange}(0, B_K)$" + "\n"
        r"• 广播合成 2D 指针网格 (利用广播规则 [:, None] 与 [None, :])：" + "\n"
        r"  $A_{\mathrm{ptr}} = A + (\mathrm{offs\_m}[:, \mathrm{None}] \cdot \mathrm{stride}_{am} + \mathrm{offs\_k}[\mathrm{None}, :] \cdot \mathrm{stride}_{ak})$" + "\n"
        r"  $B_{\mathrm{ptr}} = B + (\mathrm{offs\_k}[:, \mathrm{None}] \cdot \mathrm{stride}_{bk} + \mathrm{offs\_n}[\mathrm{None}, :] \cdot \mathrm{stride}_{bn})$" + "\n"
        r"• K 维收缩主循环步进：" + "\n"
        r"  每步仅需标量指针加法：$A_{\mathrm{ptr}} += B_K \cdot \mathrm{stride}_{ak}, \ B_{\mathrm{ptr}} += B_K \cdot \mathrm{stride}_{bk}$。"
    )
    ax.text(6, 81.5, txt_ptr, fontsize=8.8, color="#1E293B", va="top")

    # Right: PID Swizzling
    ax.add_patch(patches.FancyBboxPatch((52, 48), 44, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_dark"], linewidth=1.5))
    ax.text(74, 85, "2. Program ID 超分组 Swizzling 映射", ha="center", fontsize=11, fontweight="bold", color=PALETTE["green_dark"])
    txt_swiz = (
        "• 朴素行优先发射（Row-Major）的 L2 Cache 灾难：\n"
        r"  若按 pid_m = pid // num_pid_n 发射，连续的 Block 扫描整行输出；" + "\n"
        r"  导致矩阵 B 的所有列瓦片在 L2 Cache 中被频繁逐行刷新挤出，命中率 $< 15\%$。\n"
        "• Grouped PID Swizzling 超分组调度：\n"
        r"  定义超分组大小 $G = 8$ (GROUP_SIZE_M)；" + "\n"
        r"  每个 Super-Group 包含 $G \times \mathrm{num\_pid\_n}$ 个分块，并在组内优先按列推进：" + "\n"
        r"  $\mathrm{group\_id} = \mathrm{pid} // (G \cdot \mathrm{num\_pid\_n})$" + "\n"
        r"  $\mathrm{first\_pid\_m} = \mathrm{group\_id} \cdot G$" + "\n"
        r"  $\mathrm{group\_size\_m} = \min(\mathrm{num\_pid\_m} - \mathrm{first\_pid\_m}, G)$" + "\n"
        r"  $\mathrm{pid\_m} = \mathrm{first\_pid\_m} + (\mathrm{pid} \% \mathrm{group\_size\_m})$" + "\n"
        r"  $\mathrm{pid\_n} = (\mathrm{pid} \% (G \cdot \mathrm{num\_pid\_n})) // \mathrm{group\_size\_m}$"
    )
    ax.text(54, 81.5, txt_swiz, fontsize=8.6, color="#1E293B", va="top")

    # Bottom: Cache reuse benefit
    rect_bot = patches.FancyBboxPatch((4, 4), 92, 41, boxstyle="round,pad=0.4,rounding_size=1",
                                       facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bot)

    ax.text(6, 41, "L2 Cache 局部性提升与显存带宽节约数学证明：", fontsize=11, fontweight="bold", color="#0F172A")
    txt_cache = (
        "1. L2 缓存常驻工作集（Working Set）分析：\n"
        r"   • 朴素无 Swizzle 模式：计算一行输出需要依次读取整张矩阵 B。当 N 很大时，B 无法常驻于 L2 Cache (几十 MB)，导致矩阵 A 和 B 均反复退化为从 HBM 重新加载；" + "\n"
        r"   • 引入 Grouped Swizzle 后：硬件 SM 集中在 $G \times B_M$ 行与 $B_N$ 列的紧凑局部窗口内计算。在此窗口内，矩阵 B 的列切片被连续 $G$ 个不同的 M 块共享！" + "\n\n"
        "2. 数据复用度量化提升：\n"
        r"   矩阵 B 在 L2 Cache 中的平均命中率提升为原先的 $G$ 倍（当 $G=8$ 时，B 的片外 DRAM 读取次数骤降至近 $1/8$）。" + "\n"
        "   这一极简的索引置换代数公式，零运行时计算开销，纯粹通过空间局部性变换使 Triton GEMM 内核性能提升高达 30%~50%，逼近 cuBLAS 极限！"
    )
    ax.text(6, 37.5, txt_cache, fontsize=9.0, color="#334155", va="top")

    p = Path("triton/assets/figs/triton_matmul_pid_swizzle_math.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

if __name__ == "__main__":
    make_fig_train_sp()
    make_fig_cuda_online_softmax()
    make_fig_cuda_gemm_tiling()
    make_fig_train_ddp_ring_allreduce()
    make_fig_inference_prefill_vs_decode()
    make_fig_rl_policy_gradient()
    make_fig_cuda_deepgemm_fp8()
    make_fig_cuda_deepep_moe()
    make_fig_train_cp_ring_attention()
    make_fig_triton_matmul_pid_swizzle()
    print("All core mathematical principle diagrams generated successfully!")
