# -*- coding: utf-8 -*-
"""
Generate a clear, publication-quality mathematical equivalence diagram for 
Megatron-LM Tensor Parallelism (TP) Column-Row MLP cascade.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_tp_math_equivalence_figure(out_path="train/assets/figs/tp_mlp_math_equivalence.png"):
    # Style setup
    plt.rcParams.update({
        "font.family": ["Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
    })

    fig = plt.figure(figsize=(14.5, 8.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_gpu0 = "#1E88E5"          # Blue for GPU 0
    c_gpu0_light = "#E3F2FD"
    c_gpu1 = "#E65100"          # Orange/Amber for GPU 1
    c_gpu1_light = "#FFF3E0"
    c_shared = "#455A64"        # Slate grey for shared/replicated
    c_shared_light = "#ECEFF1"
    c_accent = "#2E7D32"        # Green for AllReduce / Final Output
    c_accent_light = "#E8F5E9"

    # Outer Canvas Frame
    rect_main = patches.FancyBboxPatch((1.5, 1.5), 97, 97, boxstyle="round,pad=0.5,rounding_size=1.2",
                                       facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.5)
    ax.add_patch(rect_main)

    # Title & Subtitle Banner
    ax.text(50, 95.5, "张量并行（TP）MLP 模块“先列后行”代数等价性与零中间通信推导", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#0F172A")
    ax.text(50, 92.5, r"数学恒等式: $\sigma(X W_1) W_2 = \mathrm{AllReduce}\left( \sigma(X W_{1, i}) W_{2, i} \right) \equiv Y$", 
            ha="center", va="center", fontsize=12, color="#334155")

    # -------------------------------------------------------------
    # Step 1: Column Parallel GEMM
    # -------------------------------------------------------------
    ax.text(4, 87.5, "步骤 1：第一层列并行 (Column Parallel GEMM)", fontsize=12, fontweight="bold", color=c_gpu0)
    ax.text(4, 84.5, "权重 $W_1$ 沿输出列维度拆分；输入 $X$ 两卡共享；各卡独立计算本地矩阵乘，得到天然列切分的中间特征：", 
            fontsize=9.8, color="#475569")

    # Tensor X
    ax.add_patch(patches.Rectangle((4, 71), 7.5, 11, facecolor=c_shared_light, edgecolor=c_shared, linewidth=1.5))
    ax.text(7.75, 76.5, "$X$\n$(B \\times d)$", ha="center", va="center", fontsize=10, fontweight="bold", color=c_shared)
    ax.text(7.75, 68.5, "输入矩阵 $X$\n(各卡持有全量)", ha="center", va="top", fontsize=8.5, color="#64748B")

    ax.text(13.2, 76.5, r"$\times$", ha="center", va="center", fontsize=16, fontweight="bold", color="#64748B")

    # Tensor W1 split vertically
    ax.add_patch(patches.Rectangle((15, 71), 6.5, 11, facecolor=c_gpu0_light, edgecolor=c_gpu0, linewidth=1.5))
    ax.text(18.25, 76.5, "$W_{1, 0}$\n$(d \\times \\frac{4d}{2})$", ha="center", va="center", fontsize=9.2, fontweight="bold", color=c_gpu0)
    ax.text(18.25, 68.5, "GPU 0 列切片", ha="center", va="top", fontsize=8.5, color=c_gpu0, fontweight="bold")

    ax.add_patch(patches.Rectangle((21.5, 71), 6.5, 11, facecolor=c_gpu1_light, edgecolor=c_gpu1, linewidth=1.5))
    ax.text(24.75, 76.5, "$W_{1, 1}$\n$(d \\times \\frac{4d}{2})$", ha="center", va="center", fontsize=9.2, fontweight="bold", color=c_gpu1)
    ax.text(24.75, 68.5, "GPU 1 列切片", ha="center", va="top", fontsize=8.5, color=c_gpu1, fontweight="bold")

    ax.add_patch(patches.Rectangle((15, 71), 13, 11, facecolor="none", edgecolor="#1E293B", linestyle="--", linewidth=1.2))

    ax.text(30, 76.5, r"$=$", ha="center", va="center", fontsize=16, fontweight="bold", color="#64748B")

    # Tensor H split vertically
    ax.add_patch(patches.Rectangle((32, 71), 7.5, 11, facecolor=c_gpu0_light, edgecolor=c_gpu0, linewidth=1.5))
    ax.text(35.75, 76.5, "$H_0 = X W_{1,0}$\n$(B \\times \\frac{4d}{2})$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_gpu0)

    ax.add_patch(patches.Rectangle((39.5, 71), 7.5, 11, facecolor=c_gpu1_light, edgecolor=c_gpu1, linewidth=1.5))
    ax.text(43.25, 76.5, "$H_1 = X W_{1,1}$\n$(B \\times \\frac{4d}{2})$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_gpu1)

    ax.add_patch(patches.Rectangle((32, 71), 15, 11, facecolor="none", edgecolor="#1E293B", linestyle="--", linewidth=1.2))
    ax.text(39.5, 68.5, "中间特征 $[H_0 \\mid H_1]$\n(各卡分别产出且仅存本地)", ha="center", va="top", fontsize=8.5, color="#64748B")

    # -------------------------------------------------------------
    # Step 2: Element-wise Activation (Key Breakthrough)
    # -------------------------------------------------------------
    ax.annotate("", xy=(51.5, 76.5), xytext=(48.5, 76.5),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=2.2))

    rect_act = patches.FancyBboxPatch((52.5, 67), 44, 20, boxstyle="round,pad=0.4,rounding_size=1",
                                     facecolor="#FEFCE8", edgecolor="#EAB308", linewidth=1.5)
    ax.add_patch(rect_act)
    ax.text(54.5, 83.5, "步骤 2：逐元素激活函数的数学穿透性 (零中间通信的核心！)", fontsize=10.8, fontweight="bold", color="#854D0E")
    ax.text(54.5, 79.5, "• 激活函数 $\\sigma$（如 GeLU / SiLU / ReLU）是逐元素标量运算 (Element-wise)。", fontsize=9.2, color="#713F12")
    ax.text(54.5, 76.2, "• 逐点运算与按列拼接操作满足严格可交换律：", fontsize=9.2, color="#713F12")
    ax.text(54.5, 72.5, r"  $\sigma([H_0 \mid H_1]) = [\sigma(H_0) \mid \sigma(H_1)] = [\widetilde{H}_0 \mid \widetilde{H}_1]$", 
            fontsize=10.2, fontweight="bold", color="#854D0E")
    ax.text(54.5, 68.5, "★ 关键结论：各 GPU 在本地计算 $\\sigma$ 即可，各维度互不干涉，通信开销严格为 0！", 
            fontsize=9.2, fontweight="bold", color="#B45309")

    # -------------------------------------------------------------
    # Step 3: Row Parallel GEMM & Block Inner Product
    # -------------------------------------------------------------
    ax.text(4, 58, "步骤 3：第二层行并行与分块矩阵乘法 (Row Parallel GEMM)", fontsize=12, fontweight="bold", color=c_gpu1)
    ax.text(4, 55, "将权重 $W_2$ 按行切分。激活切片与权重行切片在收缩维度 ($4d/2$) 尺寸完美对齐，满足分块乘法展开：", 
            fontsize=9.8, color="#475569")

    # Act tensor [H0_tilde | H1_tilde]
    ax.add_patch(patches.Rectangle((4, 38), 8, 13.5, facecolor=c_gpu0_light, edgecolor=c_gpu0, linewidth=1.5))
    ax.text(8, 44.75, r"$\widetilde{H}_0 = \sigma(H_0)$" + "\n" + r"$(B \times \frac{4d}{2})$", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_gpu0)

    ax.add_patch(patches.Rectangle((12, 38), 8, 13.5, facecolor=c_gpu1_light, edgecolor=c_gpu1, linewidth=1.5))
    ax.text(16, 44.75, r"$\widetilde{H}_1 = \sigma(H_1)$" + "\n" + r"$(B \times \frac{4d}{2})$", 
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_gpu1)

    ax.add_patch(patches.Rectangle((4, 38), 16, 13.5, facecolor="none", edgecolor="#1E293B", linestyle="--", linewidth=1.2))
    ax.text(12, 35, "激活值按列分片\n$[\widetilde{H}_0 \\mid \\widetilde{H}_1]$", ha="center", va="top", fontsize=8.5, color="#64748B")

    ax.text(21.5, 44.75, r"$\times$", ha="center", va="center", fontsize=18, fontweight="bold", color="#64748B")

    # W2 tensor split horizontally
    # GPU 0: top row
    ax.add_patch(patches.Rectangle((23.5, 45.25), 14, 6.25, facecolor=c_gpu0_light, edgecolor=c_gpu0, linewidth=1.5))
    ax.text(30.5, 48.37, "$W_{2, 0}$ (GPU 0 行切片) $(\\frac{4d}{2} \\times d)$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_gpu0)

    # GPU 1: bottom row
    ax.add_patch(patches.Rectangle((23.5, 38), 14, 6.25, facecolor=c_gpu1_light, edgecolor=c_gpu1, linewidth=1.5))
    ax.text(30.5, 41.12, "$W_{2, 1}$ (GPU 1 行切片) $(\\frac{4d}{2} \\times d)$", ha="center", va="center", fontsize=8.8, fontweight="bold", color=c_gpu1)

    ax.add_patch(patches.Rectangle((23.5, 38), 14, 13.5, facecolor="none", edgecolor="#1E293B", linestyle="--", linewidth=1.2))
    ax.text(30.5, 35, "权重按行分片\n$[W_{2,0} ; W_{2,1}]$", ha="center", va="top", fontsize=8.5, color="#64748B")

    ax.text(39.5, 44.75, r"$=$", ha="center", va="center", fontsize=18, fontweight="bold", color="#64748B")

    # Partial products addition
    # Z0
    ax.add_patch(patches.Rectangle((41.5, 40.75), 8.5, 8, facecolor=c_gpu0_light, edgecolor=c_gpu0, linewidth=1.5))
    ax.text(45.75, 44.75, "$Z_0 = \\widetilde{H}_0 W_{2,0}$\n$(B \\times d)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_gpu0)
    ax.text(45.75, 37.5, "GPU 0 局部积", ha="center", va="top", fontsize=8.5, color=c_gpu0, fontweight="bold")

    ax.text(51.5, 44.75, r"$+$", ha="center", va="center", fontsize=18, fontweight="bold", color="#2E7D32")

    # Z1
    ax.add_patch(patches.Rectangle((53, 40.75), 8.5, 8, facecolor=c_gpu1_light, edgecolor=c_gpu1, linewidth=1.5))
    ax.text(57.25, 44.75, "$Z_1 = \\widetilde{H}_1 W_{2,1}$\n$(B \\times d)$", ha="center", va="center", fontsize=8.5, fontweight="bold", color=c_gpu1)
    ax.text(57.25, 37.5, "GPU 1 局部积", ha="center", va="top", fontsize=8.5, color=c_gpu1, fontweight="bold")

    # -------------------------------------------------------------
    # Step 4: AllReduce (Sum)
    # -------------------------------------------------------------
    ax.annotate("", xy=(65, 44.75), xytext=(62.5, 44.75),
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.5))

    # AllReduce Box
    rect_ar = patches.FancyBboxPatch((65.5, 38.5), 13.5, 12.5, boxstyle="round,pad=0.4,rounding_size=1",
                                     facecolor=c_accent_light, edgecolor=c_accent, linewidth=1.8)
    ax.add_patch(rect_ar)
    ax.text(72.25, 46.5, "AllReduce\n(SUM 同步)", ha="center", va="center", fontsize=10.8, fontweight="bold", color=c_accent)
    ax.text(72.25, 41, "跨卡累加求和", ha="center", va="center", fontsize=8.8, color="#1B5E20")

    ax.annotate("", xy=(82, 44.75), xytext=(79.5, 44.75),
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=2.5))

    # Final Output Y
    ax.add_patch(patches.Rectangle((82.5, 39.5), 14, 10.5, facecolor="#E8F5E9", edgecolor="#1B5E20", linewidth=2.0))
    ax.text(89.5, 44.75, r"$\mathbf{Y} = Z_0 + Z_1$" + "\n" + r"$(B \times d)$" + "\n" + r"$\equiv \sigma(X W_1) W_2$", 
            ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1B5E20")
    ax.text(89.5, 37.5, "★ 严谨代数等价全局输出\n(同步后两卡各持有一份)", ha="center", va="top", fontsize=8.5, color="#1B5E20", fontweight="bold")

    # -------------------------------------------------------------
    # Bottom Mathematical Invariant & Contrast Summary
    # -------------------------------------------------------------
    rect_bottom = patches.FancyBboxPatch((4, 3.5), 92.5, 23.5, boxstyle="round,pad=0.4,rounding_size=1",
                                         facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=1.2)
    ax.add_patch(rect_bottom)

    ax.text(6, 23.5, "底层数学原理总结与为什么“先行后列”必然失败（反证）：", fontsize=11, fontweight="bold", color="#0F172A")

    col1_txt = (
        "1. 为什么“先列后行”等价于原函数？\n"
        "   • 线性代数分块恒等式：设行块向量 $A=[A_1, A_2]$ 与列块向量 $B=[B_1 ; B_2]$，则内积乘积展开为 $A B = A_1 B_1 + A_2 B_2$。\n"
        "   • 第一层列并行输出 $[H_0 \\mid H_1]$ 天然构成了第二层 GEMM 左侧的分块行向量；\n"
        "   • 激活函数 $\\sigma$ 是逐元素的（$\\sigma([H_0 \\mid H_1]) = [\\sigma(H_0) \\mid \\sigma(H_1)]$），对切分透明，无需跨卡交换任何数据；\n"
        "   • 第二层行并行中，GPU 0 计算 $Z_0=\\widetilde{H}_0 W_{2,0}$，GPU 1 计算 $Z_1=\\widetilde{H}_1 W_{2,1}$，刚好分别对应分块乘法中的两项加数！\n"
        "   • 整个 MLP 两层 GEMM 中间零通信，最终仅需一次全卡 AllReduce(Sum) 即可完成 $Z_0 + Z_1$，实现与原始单卡计算的完全等价。"
    )
    ax.text(6, 21.2, col1_txt, fontsize=8.8, color="#334155", va="top")

    col2_txt = (
        "2. 为什么不能反过来采用“先行后列”？（为什么先行后列通信量翻倍？）\n"
        "   • 若第一层采用行并行，各卡计算部分和：$H = X_0 W_{1,0} + X_1 W_{1,1}$。\n"
        "   • 关键矛盾：非线性激活函数对加法不满足分配律，即 $\\sigma(A + B) \\neq \\sigma(A) + \\sigma(B)$！各卡无法在本地先执行 $\\sigma$！\n"
        "   • 因此第一层后必须先做一次 AllReduce 得到完整的 $H$，才能计算 $\\sigma(H)$；紧接着第二层列并行又需要后续通信拼接——通信开销直接翻倍！"
    )
    ax.text(6, 8.8, col2_txt, fontsize=8.8, color="#991B1B", va="top")

    # Save
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated successfully: {p}")

if __name__ == "__main__":
    generate_tp_math_equivalence_figure()
