# -*- coding: utf-8 -*-
"""
Scientific Figure Generator based on scientific-figure-making skill.
Produces publication-ready high-DPI figures for HPC Learning courses.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Canonical Palette from scientific-figure-making
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "neutral_dark": "#4D4D4D",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
}

def apply_publication_style(font_size=14, linewidth=2.0):
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": font_size,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": linewidth,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

def finalize_figure(fig, out_path, dpi=300):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.5)
    fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {p}")

# -------------------------------------------------------------
# 1. Training Track Figures
# -------------------------------------------------------------
def make_train_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. ZeRO Memory Breakdown
    fig, ax = plt.subplots(figsize=(8, 5))
    stages = ["Standard\n(ZeRO-0)", "ZeRO-1\n(Opt State)", "ZeRO-2\n(+ Grad)", "ZeRO-3\n(+ Param)"]
    params = np.array([16.0, 16.0, 16.0, 2.0])      # 16GB total -> 2GB on 8 GPUs
    grads = np.array([16.0, 16.0, 2.0, 2.0])
    opt_states = np.array([48.0, 6.0, 6.0, 6.0])     # 48GB FP32 Adam -> 6GB on 8 GPUs
    activations = np.array([14.0, 14.0, 14.0, 14.0])
    
    x = np.arange(len(stages))
    w = 0.55
    
    p1 = ax.bar(x, params, w, label="Parameters (FP16)", color=PALETTE["blue_secondary"], edgecolor="black", linewidth=1.2)
    p2 = ax.bar(x, grads, w, bottom=params, label="Gradients (FP16)", color=PALETTE["teal"], edgecolor="black", linewidth=1.2)
    p3 = ax.bar(x, opt_states, w, bottom=params+grads, label="Optimizer States (Adam FP32)", color=PALETTE["red_strong"], edgecolor="black", linewidth=1.2)
    p4 = ax.bar(x, activations, w, bottom=params+grads+opt_states, label="Residual Activations", color=PALETTE["green_3"], edgecolor="black", linewidth=1.2)
    
    total = params + grads + opt_states + activations
    for i, tot in enumerate(total):
        ax.text(i, tot + 1.5, f"{tot:.0f} GB", ha="center", va="bottom", fontweight="bold", fontsize=12)
        
    ax.set_ylabel("Per-GPU Memory Footprint (GB)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("ZeRO Memory Breakdown Across Stages (7B Model, 8x GPUs)", pad=15, fontweight="bold", fontsize=14)
    finalize_figure(fig, "train/assets/figs/train_zero_memory.png")

    # 2. Collective Communication Volume
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ranks = np.array([2, 4, 8, 16, 32, 64])
    allreduce_vol = 2.0 * (ranks - 1) / ranks
    ax.plot(ranks, allreduce_vol, "o-", color=PALETTE["blue_main"], linewidth=2.5, markersize=7, label="Ring AllReduce Factor: 2(p-1)/p")
    ax.axhline(2.0, linestyle="--", color=PALETTE["red_strong"], alpha=0.7, label="Theoretical Upper Bound (2.0x)")
    ax.set_xlabel("Number of Distributed GPUs (Ranks)", fontweight="bold")
    ax.set_ylabel("Normalized Communication Volume / Data Size", fontweight="bold")
    ax.set_title("Ring Collective Communication Scalability", pad=15, fontweight="bold")
    ax.set_ylim(0.8, 2.3)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11)
    finalize_figure(fig, "train/assets/figs/train_comm_volume.png")

    # 3. Pipeline Parallelism Idle Bubble
    fig, ax = plt.subplots(figsize=(8, 4.8))
    m = np.linspace(4, 64, 100)
    for p_stages, col, lbl in [(4, PALETTE["teal"], "4 Pipeline Stages"), (8, PALETTE["blue_main"], "8 Pipeline Stages"), (16, PALETTE["red_strong"], "16 Pipeline Stages")]:
        bubble_frac = (p_stages - 1) / (m + p_stages - 1) * 100.0
        ax.plot(m, bubble_frac, linewidth=2.5, color=col, label=lbl)
    ax.set_xlabel("Number of Microbatches (m)", fontweight="bold")
    ax.set_ylabel("Idle Bubble Fraction (%)", fontweight="bold")
    ax.set_title("1F1B Pipeline Parallelism Bubble Ratio: (p-1)/(m+p-1)", pad=15, fontweight="bold")
    ax.set_ylim(0, 75)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=11)
    finalize_figure(fig, "train/assets/figs/train_pp_bubble.png")

# -------------------------------------------------------------
# 2. CUDA Track Figures
# -------------------------------------------------------------
def make_cuda_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. Roofline Model
    fig, ax = plt.subplots(figsize=(8.5, 5))
    intensity = np.logspace(-1, 3, 200)
    peak_flops = 989.0  # TFLOPS H100 FP16 TC
    peak_bw = 3.35      # TB/s H100 SXM5 HBM3
    
    achievable = np.minimum(peak_flops, intensity * peak_bw)
    turning_pt = peak_flops / peak_bw
    
    ax.loglog(intensity, achievable, color=PALETTE["blue_main"], linewidth=3, label="H100 Roofline (FP16 Tensor Core)")
    ax.axvline(turning_pt, linestyle="--", color=PALETTE["red_strong"], linewidth=1.5, alpha=0.8, label=f"Machine Balance: {turning_pt:.1f} FLOP/Byte")
    
    # Annotate regions
    ax.text(0.5, 4.0, "Memory Bandwidth Bound\n(Softmax, LayerNorm, RMSNorm)", color=PALETTE["red_strong"], fontweight="bold", fontsize=11)
    ax.text(60, 400.0, "Compute Bound\n(Tiled GEMM, Conv)", color=PALETTE["blue_main"], fontweight="bold", fontsize=11)
    
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontweight="bold")
    ax.set_ylabel("Attainable Performance (TFLOPS)", fontweight="bold")
    ax.set_title("NVIDIA H100 Roofline Model & Operational Regimes", pad=15, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11)
    finalize_figure(fig, "cuda/assets/figs/cuda_roofline.png")

    # 2. Shared Memory Bank Conflict Stride
    fig, ax = plt.subplots(figsize=(8, 4.8))
    strides = np.array([1, 2, 4, 8, 16, 32])
    ways = np.array([1, 2, 4, 8, 16, 32]) # Bank conflicts
    throughput = 100.0 / ways
    
    bars = ax.bar([str(s) for s in strides], throughput, width=0.5, color=PALETTE["blue_secondary"], edgecolor="black", linewidth=1.2)
    bars[0].set_color(PALETTE["green_3"])
    bars[0].set_edgecolor("black")
    bars[-1].set_color(PALETTE["red_strong"])
    bars[-1].set_edgecolor("black")
    
    for b, w in zip(bars, ways):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{w}-way conflict\n({b.get_height():.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    ax.set_xlabel("Access Stride (32-bit floats)", fontweight="bold")
    ax.set_ylabel("Effective Shared Memory Throughput (%)", fontweight="bold")
    ax.set_title("Shared Memory Bank Conflict Degradation with Power-of-2 Strides", pad=15, fontweight="bold")
    ax.set_ylim(0, 125)
    finalize_figure(fig, "cuda/assets/figs/cuda_smem_bank_conflict.png")

    # 3. Warp Shuffle vs SMEM Latency
    fig, ax = plt.subplots(figsize=(8, 4.8))
    methods = ["Shared Memory\n(__syncthreads + Bank Load)", "Warp Shuffle\n(__shfl_down_sync)"]
    latencies = [42.0, 8.5] # Clock cycles
    colors = [PALETTE["neutral_dark"], PALETTE["blue_main"]]
    
    b = ax.bar(methods, latencies, width=0.45, color=colors, edgecolor="black", linewidth=1.2)
    for rect in b:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 1.2, f"{h:.1f} cycles", ha="center", va="bottom", fontweight="bold", fontsize=12)
        
    ax.set_ylabel("Latency (GPU Clock Cycles)", fontweight="bold")
    ax.set_title("Warp Shuffle vs. Shared Memory Reduction Latency", pad=15, fontweight="bold")
    ax.set_ylim(0, 55)
    finalize_figure(fig, "cuda/assets/figs/cuda_warp_shuffle_latency.png")

# -------------------------------------------------------------
# 3. Triton Track Figures
# -------------------------------------------------------------
def make_triton_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. Fused Softmax vs PyTorch Native
    fig, ax = plt.subplots(figsize=(8, 4.8))
    seq_lens = np.array([512, 1024, 2048, 4096, 8192])
    torch_latency = np.array([0.45, 0.88, 1.82, 3.85, 8.10])
    triton_latency = np.array([0.12, 0.22, 0.46, 0.94, 1.95])
    
    ax.plot(seq_lens, torch_latency, "s--", color=PALETTE["red_strong"], linewidth=2.5, markersize=7, label="PyTorch Native (Multiple HBM Roundtrips)")
    ax.plot(seq_lens, triton_latency, "o-", color=PALETTE["blue_main"], linewidth=2.5, markersize=7, label="Triton Fused Kernel (SRAM Single-Pass)")
    
    for x, y_t, y_tr in zip(seq_lens, torch_latency, triton_latency):
        speedup = y_t / y_tr
        ax.annotate(f"{speedup:.1f}x", xy=(x, y_tr), xytext=(x, y_tr * 0.55),
                    ha="center", fontsize=10, fontweight="bold", color=PALETTE["blue_main"])
                    
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length (Tokens)", fontweight="bold")
    ax.set_ylabel("Execution Time (ms, log scale)", fontweight="bold")
    ax.set_title("Softmax Kernel Execution Latency: Triton vs PyTorch Eager", pad=15, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    finalize_figure(fig, "triton/assets/figs/triton_fused_softmax_speedup.png")

# -------------------------------------------------------------
# 4. Inference Track Figures
# -------------------------------------------------------------
def make_inference_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. KV Cache Memory Growth
    fig, ax = plt.subplots(figsize=(8.5, 5))
    seq_len = np.linspace(512, 32768, 100)
    # Llama-3 8B: 32 layers, 8 KV heads, 128 head dim, FP16 = 2 * 32 * 8 * 128 * 2 = 131,072 bytes per token
    bytes_per_token = 2 * 32 * 8 * 128 * 2
    
    for bs, col, lbl in [(1, PALETTE["teal"], "Batch Size = 1"),
                         (8, PALETTE["green_3"], "Batch Size = 8"),
                         (32, PALETTE["blue_main"], "Batch Size = 32"),
                         (64, PALETTE["red_strong"], "Batch Size = 64")]:
        vram_gb = (seq_len * bytes_per_token * bs) / (1024**3)
        ax.plot(seq_len / 1024, vram_gb, linewidth=2.5, color=col, label=lbl)
        
    ax.axhline(80, linestyle="--", color=PALETTE["neutral_dark"], alpha=0.7, label="80GB VRAM Limit (Single GPU)")
    ax.set_xlabel("Sequence Length (K Tokens)", fontweight="bold")
    ax.set_ylabel("KV Cache Memory Footprint (GB)", fontweight="bold")
    ax.set_title("Llama-3 8B KV Cache Scaling with Context & Concurrency", pad=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    finalize_figure(fig, "inference/assets/figs/kvcache_memory_growth.png")

    # 2. Speculative Decoding Speedup
    fig, ax = plt.subplots(figsize=(8, 4.8))
    alpha = np.linspace(0.4, 0.95, 100) # Acceptance rate
    for gamma, col, lbl in [(3, PALETTE["teal"], "Draft K = 3 tokens"),
                            (5, PALETTE["blue_main"], "Draft K = 5 tokens"),
                            (7, PALETTE["violet"], "Draft K = 7 tokens")]:
        # Speedup formula ~ (1 - alpha^(K+1)) / ((1 - alpha) * (1 + c * K))
        c = 0.08 # cost ratio draft / target
        speedup = (1.0 - alpha**(gamma + 1)) / ((1.0 - alpha) * (1.0 + c * gamma))
        ax.plot(alpha, speedup, linewidth=2.5, color=col, label=lbl)
        
    ax.axhline(1.0, linestyle="--", color="black", alpha=0.5, label="Baseline (1.0x)")
    ax.set_xlabel("Token Acceptance Rate (α)", fontweight="bold")
    ax.set_ylabel("Theoretical Wall-Clock Speedup", fontweight="bold")
    ax.set_title("Speculative Decoding Speedup vs. Acceptance Rate", pad=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    finalize_figure(fig, "inference/assets/figs/speculative_speedup_model.png")

    # 3. Quantization Tradeoff
    fig, ax = plt.subplots(figsize=(8, 4.8))
    formats = ["FP16", "FP8 (E4M3)", "INT4 (AWQ/GPTQ)"]
    weights_gb = [14.0, 7.0, 3.5]
    throughput_toks = [85.0, 155.0, 240.0]
    
    x = np.arange(len(formats))
    w = 0.35
    
    ax1 = ax
    ax2 = ax.twinx()
    
    b1 = ax1.bar(x - w/2, weights_gb, w, label="Weight Memory (GB)", color=PALETTE["blue_secondary"], edgecolor="black", linewidth=1.2)
    b2 = ax2.bar(x + w/2, throughput_toks, w, label="Decode Throughput (Tokens/s)", color=PALETTE["green_3"], edgecolor="black", linewidth=1.2)
    
    ax1.set_ylabel("Weights Footprint (GB)", color=PALETTE["blue_main"], fontweight="bold")
    ax2.set_ylabel("Serving Throughput (Tokens/s)", color="#2D7F2D", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(formats, fontweight="bold")
    ax1.set_ylim(0, 20)
    ax2.set_ylim(0, 300)
    ax.set_title("Weight Quantization: Memory Footprint vs Serving Throughput (7B Model)", pad=15, fontweight="bold")
    finalize_figure(fig, "inference/assets/figs/quantization_tradeoff.png")

# -------------------------------------------------------------
# 5. RL Track Figures
# -------------------------------------------------------------
def make_rl_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. Bias-Variance Tradeoff in GAE
    fig, ax = plt.subplots(figsize=(8, 4.8))
    lam = np.linspace(0, 1, 100)
    bias = (1.0 - lam)**2 * 10.0
    variance = lam**2 * 10.0
    total_error = bias + variance
    
    ax.plot(lam, bias, "--", color=PALETTE["red_strong"], linewidth=2.2, label="Bias (High at λ=0: TD(0))")
    ax.plot(lam, variance, "-.", color=PALETTE["teal"], linewidth=2.2, label="Variance (High at λ=1: MC)")
    ax.plot(lam, total_error, "-", color=PALETTE["blue_main"], linewidth=3.0, label="Total Error (Optimal λ ≈ 0.95)")
    
    opt_idx = np.argmin(total_error)
    ax.axvline(lam[opt_idx], color=PALETTE["highlight"], linestyle=":", linewidth=2, label=f"Sweet Spot (λ={lam[opt_idx]:.2f})")
    
    ax.set_xlabel("GAE Decay Factor (λ)", fontweight="bold")
    ax.set_ylabel("Normalized Estimation Error", fontweight="bold")
    ax.set_title("Generalized Advantage Estimation (GAE) Bias-Variance Tradeoff", pad=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper center", fontsize=10)
    finalize_figure(fig, "rl/assets/figs/rl_bias_variance_tradeoff.png")

    # 2. PPO Surrogate Objective Curves
    fig, ax = plt.subplots(figsize=(8, 4.8))
    r = np.linspace(0.6, 1.4, 200)
    eps = 0.2
    
    # Advantage > 0
    surr1_pos = r * 1.0
    surr2_pos = np.clip(r, 1 - eps, 1 + eps) * 1.0
    l_clip_pos = np.minimum(surr1_pos, surr2_pos)
    
    # Advantage < 0
    surr1_neg = r * (-1.0)
    surr2_neg = np.clip(r, 1 - eps, 1 + eps) * (-1.0)
    l_clip_neg = np.minimum(surr1_neg, surr2_neg)
    
    ax.plot(r, l_clip_pos, color=PALETTE["blue_main"], linewidth=2.5, label="Objective L_CLIP (Advantage A > 0)")
    ax.plot(r, l_clip_neg, color=PALETTE["red_strong"], linewidth=2.5, label="Objective L_CLIP (Advantage A < 0)")
    ax.axvline(1 - eps, linestyle=":", color="gray", label=f"Clip Boundaries: 1 ± ε ({1-eps:.1f}, {1+eps:.1f})")
    ax.axvline(1 + eps, linestyle=":", color="gray")
    
    ax.set_xlabel("Probability Ratio r_t(θ) = π_θ(a|s) / π_old(a|s)", fontweight="bold")
    ax.set_ylabel("Pessimistic Clipped Objective Value", fontweight="bold")
    ax.set_title("PPO-Clip Objective Function & Conservative Bounds (ε=0.2)", pad=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)
    finalize_figure(fig, "rl/assets/figs/ppo_surrogate_objective.png")

    # 3. GRPO Group Relative Advantage
    fig, ax = plt.subplots(figsize=(8, 4.8))
    groups = [f"Output {i+1}" for i in range(8)]
    raw_rewards = np.array([0.0, 0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 1.0])
    mean_r = np.mean(raw_rewards)
    std_r = np.std(raw_rewards) + 1e-8
    norm_adv = (raw_rewards - mean_r) / std_r
    
    colors = [PALETTE["red_strong"] if a < 0 else PALETTE["green_3"] if a > 0 else PALETTE["neutral"] for a in norm_adv]
    bars = ax.bar(groups, norm_adv, width=0.5, color=colors, edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=1)
    
    for b, a in zip(bars, norm_adv):
        val = f"{a:+.2f}"
        ax.text(b.get_x() + b.get_width()/2, a + (0.08 if a >= 0 else -0.15), val, ha="center", va="bottom", fontweight="bold", fontsize=10)
        
    ax.set_ylabel("Group Relative Advantage A_i", fontweight="bold")
    ax.set_title("GRPO Zero-Critic Group Relative Advantage Normalization (G=8)", pad=15, fontweight="bold")
    ax.set_ylim(-2.0, 2.0)
    finalize_figure(fig, "rl/assets/figs/grpo_reward_distribution.png")

# -------------------------------------------------------------
# 6. Runtime Track Figures
# -------------------------------------------------------------
def make_runtime_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. PyTorch Eager vs torch.compile
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    benchmarks = ["Pointwise / GELU", "LayerNorm Fusion", "Self-Attention", "Full Transformer Block"]
    eager = [1.0, 1.0, 1.0, 1.0] # normalized baseline
    compile_speedup = [2.85, 2.15, 1.45, 1.72]
    
    x = np.arange(len(benchmarks))
    w = 0.35
    
    ax.bar(x - w/2, eager, w, label="PyTorch Eager (1.0x)", color=PALETTE["neutral_dark"], edgecolor="black", linewidth=1.2)
    b2 = ax.bar(x + w/2, compile_speedup, w, label="torch.compile (Inductor Fusion)", color=PALETTE["blue_main"], edgecolor="black", linewidth=1.2)
    
    for rect, sp in zip(b2, compile_speedup):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.08, f"{sp:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=11)
        
    ax.set_ylabel("Normalized Speedup", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontweight="bold")
    ax.set_ylim(0, 3.5)
    ax.set_title("torch.compile Inductor Speedup vs. Eager Execution", pad=15, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    finalize_figure(fig, "runtime/assets/figs/pytorch_eager_vs_compile.png")

    # 2. Ring vs Tree AllReduce Latency Model
    fig, ax = plt.subplots(figsize=(8, 4.8))
    size_kb = np.logspace(0, 6, 200) # 1KB to 1GB
    alpha_lat = 5e-6 # 5 us
    beta_bw = 200e9  # 200 GB/s bus
    p = 64 # 64 ranks
    
    ring_time = 2 * (p - 1) * alpha_lat + (2 * (p - 1) / p) * (size_kb * 1024) / beta_bw
    tree_time = 2 * np.log2(p) * alpha_lat + (2 * np.log2(p)) * (size_kb * 1024) / beta_bw
    
    ax.loglog(size_kb, ring_time * 1e3, color=PALETTE["blue_main"], linewidth=2.5, label="Ring AllReduce: 2(p-1)α + 2(p-1)/p · S/β")
    ax.loglog(size_kb, tree_time * 1e3, color=PALETTE["red_strong"], linewidth=2.5, label="Tree AllReduce: 2·log2(p)α + 2·log2(p) · S/β")
    
    ax.set_xlabel("Message Size (KB, log scale)", fontweight="bold")
    ax.set_ylabel("Latency (ms, log scale)", fontweight="bold")
    ax.set_title("Collective Communication Analytical Latency: Ring vs Tree (p=64)", pad=15, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    finalize_figure(fig, "runtime/assets/figs/ring_vs_tree_latency.png")

# -------------------------------------------------------------
# 7. Platform Track Figures
# -------------------------------------------------------------
def make_platform_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. Daly Optimal Checkpoint Interval
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mtbf_hours = np.linspace(2, 48, 150) # Cluster MTBF from 2h to 48h
    
    for delta_min, col, lbl in [(5, PALETTE["teal"], "Save Cost δ = 5 mins"),
                                (15, PALETTE["blue_main"], "Save Cost δ = 15 mins"),
                                (30, PALETTE["red_strong"], "Save Cost δ = 30 mins")]:
        # tau = sqrt(2 * delta * MTBF)
        tau_min = np.sqrt(2 * delta_min * (mtbf_hours * 60))
        ax.plot(mtbf_hours, tau_min, linewidth=2.5, color=col, label=lbl)
        
    ax.set_xlabel("Cluster Mean Time Between Failures MTBF (Hours)", fontweight="bold")
    ax.set_ylabel("Optimal Checkpoint Interval τ (Minutes)", fontweight="bold")
    ax.set_title("Daly's Analytical Optimal Checkpoint Interval: τ = √(2 · δ · MTBF)", pad=15, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    finalize_figure(fig, "platform/assets/figs/daly_optimal_checkpoint.png")

    # 2. GPUDirect RDMA vs Host TCP Throughput
    fig, ax = plt.subplots(figsize=(8, 4.8))
    msg_size_mb = np.array([0.064, 0.256, 1.0, 4.0, 16.0, 64.0])
    host_tcp = np.array([4.2, 12.8, 28.5, 45.2, 52.0, 54.5])
    gpudirect = np.array([18.5, 62.0, 185.0, 360.0, 385.0, 392.0]) # 400Gbps NDR IB
    
    ax.plot(msg_size_mb, gpudirect, "o-", color=PALETTE["blue_main"], linewidth=2.5, markersize=7, label="GPUDirect RDMA (Zero-Copy P2P DMA)")
    ax.plot(msg_size_mb, host_tcp, "s--", color=PALETTE["red_strong"], linewidth=2.5, markersize=7, label="Host TCP/IP (CPU & RAM Staging)")
    
    ax.set_xscale("log")
    ax.set_xlabel("Message Buffer Size (MB, log scale)", fontweight="bold")
    ax.set_ylabel("Effective Network Bandwidth (Gbps)", fontweight="bold")
    ax.set_title("GPUDirect RDMA vs. Host TCP/IP Bandwidth Saturation (400Gbps IB)", pad=15, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11)
    finalize_figure(fig, "platform/assets/figs/gpudirect_vs_host_tcp.png")

# -------------------------------------------------------------
# 8. MLIR Track Figures
# -------------------------------------------------------------
def make_mlir_figures():
    apply_publication_style(font_size=13, linewidth=2.0)
    
    # 1. Compilation Time vs Speedup
    fig, ax = plt.subplots(figsize=(8, 4.8))
    stages = ["Python Eager", "TorchScript", "MLIR (Linalg)", "MLIR (LLVM/PTX)"]
    speedup = [1.0, 1.35, 2.45, 3.80]
    
    bars = ax.bar(stages, speedup, width=0.5, color=PALETTE["blue_secondary"], edgecolor="black", linewidth=1.2)
    bars[-1].set_color(PALETTE["blue_main"])
    
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.08, f"{h:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=11)
        
    ax.set_ylabel("Execution Speedup (Normalized)", fontweight="bold")
    ax.set_ylim(0, 4.5)
    ax.set_title("Performance Speedup Across Progressive Lowering Tiers", pad=15, fontweight="bold")
    finalize_figure(fig, "mlir/assets/figs/mlir_lowering_latency.png")

    # 2. Bufferization Memory Footprint
    fig, ax = plt.subplots(figsize=(8, 4.8))
    scenarios = ["Naive Out-of-Place\n(Every Op Allocates)", "One-Shot Bufferize\n(In-Place Reuse)"]
    mem_mb = [1280.0, 320.0]
    
    bars = ax.bar(scenarios, mem_mb, width=0.45, color=[PALETTE["red_strong"], PALETTE["green_3"]], edgecolor="black", linewidth=1.2)
    for b, m in zip(bars, mem_mb):
        ax.text(b.get_x() + b.get_width()/2, m + 25, f"{m:.0f} MB (-75%)" if m < 1000 else f"{m:.0f} MB", ha="center", va="bottom", fontweight="bold", fontsize=12)
        
    ax.set_ylabel("Peak Buffer Memory Footprint (MB)", fontweight="bold")
    ax.set_ylim(0, 1500)
    ax.set_title("One-Shot Bufferize Memory Reduction via In-Place Aliasing", pad=15, fontweight="bold")
    finalize_figure(fig, "mlir/assets/figs/mlir_bufferization_footprint.png")

if __name__ == "__main__":
    print("Generating Scientific Figures with scientific-figure-making skill...")
    make_train_figures()
    make_cuda_figures()
    make_triton_figures()
    make_inference_figures()
    make_rl_figures()
    make_runtime_figures()
    make_platform_figures()
    make_mlir_figures()
    print("All scientific figures successfully generated!")
