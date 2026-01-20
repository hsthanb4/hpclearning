# 在 RTX 5090 上实现 Flash Attention（CUDA C++）

> **原文链接**：https://gau-nernst.github.io/fa-5090/
> **作者**：gau-nernst
> **译者注**：本文详细记录了在 NVIDIA RTX 5090 GPU 上使用 CUDA C++ 实现 Flash Attention 的过程，从基础版本（68.20% 理论峰值性能）优化到最终版本（94.39% 速度极限），接近 CuDNN 的 97.19% 性能。

---

## 目录

1. [简介与动机](#简介与动机)
2. [前置知识](#前置知识)
3. [算法基础](#算法基础)
4. [版本 1：基础实现](#版本-1基础实现)
5. [版本 2：Shared Memory Swizzling](#版本-2shared-memory-swizzling)
6. [版本 3：2 阶段流水线](#版本-32-阶段流水线)
7. [版本 4：ldmatrix.x4 优化](#版本-4ldmatrixx4-优化)
8. [版本 5：优化的流水线](#版本-5优化的流水线)
9. [性能对比与分析](#性能对比与分析)
10. [总结与未来方向](#总结与未来方向)

---

## 简介与动机

我写这篇博客是为了记录我学习在 NVIDIA 的 RTX 5090 GPU 上使用 CUDA C++ 实现 Flash Attention 的过程。我的主要动机源于现有框架的局限性：**"许多功能在 Triton 中不可用，例如用于 sm120 的 MXFP8 / NVFP4 MMA"**。

我还注意到现有资源存在空白：虽然 **"有很多关于编写快速 matmul kernel 的优秀博客文章，但没有关于 attention 的"**。这篇文章旨在通过详细的解释和完整的开源实现来填补这一教育空白。

**实现成果**：
- 最终性能：**197.74 TFLOPS**（理论峰值的 **94.39%**）
- 起始性能：142.87 TFLOPS（68.20% SOL）
- 完整实现代码开源在 GitHub

---

## 前置知识

**强烈建议**读者熟悉以下内容：
- CUDA C++ 编程
- 如何在 NVIDIA GPU 上使用 Tensor Cores

如果需要基础知识，作者建议参考：
- **GPU-MODE 讲座系列**
- 现有的 matmul 优化资源

不需要有 attention kernel 实现经验。

---

## 算法基础

### Flash Attention 2 原理

本实现基于 Flash Attention 2 的设计原则。核心思想是将计算进行分区，使得：

> **"每个 threadblock 负责 Q 的一个块（chunk），我们将沿着 KV 的序列长度进行迭代"**

### Attention 计算的基本流程

标准的 attention 计算遵循以下模式：

```
1. Query-Key 矩阵乘法 → 产生注意力分数（attention scores）
2. Softmax 归一化
3. Attention-Value 矩阵乘法 → 生成输出
```

数学公式：
```
S = Q @ K^T
P = softmax(S)
O = P @ V
```

### 在线 Softmax（Online Softmax）

架构创新采用了**在线 softmax**算法，这是一种能够进行单次遍历 attention 计算的算法，通过维护运行统计信息：

- **最大值**（`m`）：当前看到的最大值
- **累积指数和**（`sumexp`）：归一化因子
- **未归一化的输出**（`tile_O`）：累积输出

这些统计信息可以在处理新的 key-value tiles 时高效地更新（关联性操作）。

### 分块策略

```
Q 分块：每个 threadblock 处理 [BLOCK_Q × head_dim]
KV 迭代：沿序列长度处理 [BLOCK_KV × head_dim] 的 tiles
```

**关键优势**：
- Q 数据在整个 kernel 执行期间保持在寄存器中
- 利用 head_dim 通常较小（如 128）的特性

---

## 版本 1：基础实现

### 性能指标

- **吞吐量**：142.87 TFLOPS
- **SOL 效率**：68.20%
- **配置**：`BLOCK_Q=128`, `BLOCK_KV=64`, `DIM=128`, 4 个 warps

### 内存传输架构

#### 1. Global Memory → Shared Memory

使用 **`cp.async.cg.shared.global`** PTX 指令：
- 每个线程传输 16 字节
- 处理连续的 8 个 BF16 元素
- 系统确保通过线程级组织实现合并访存

![Global to Shared Memory 数据传输](https://gau-nernst.github.io/fa-5090/global_to_shared.svg)
*图：2D tile 的 Global 到 Shared Memory 数据传输*

![合并访存模式](https://gau-nernst.github.io/fa-5090/coalesced.svg)
*图：连续线程处理连续的 8×BF16 元素组*

**函数签名**：
```cpp
template<int HEIGHT, int WIDTH, int TB_SIZE>
void global_to_shared(bf16* gmem, bf16* smem);
```

#### 2. Shared Memory → Register

使用 **`ldmatrix`** 指令，针对 MMA tile 需求：

- **Query 数据**：`ldmatrix.x4` 用于 16×16 tiles（作为乘数 A）
- **Key/Value 数据**：`ldmatrix.x2` 用于 8×16 tiles（作为乘数 B）

![Flash Attention Warp 分区](https://gau-nernst.github.io/fa-5090/fa_warp_partition.svg)
*图：Flash Attention 2 中 warps 如何分区 BLOCK_Q 维度*

![ldmatrix Tile 顺序](https://gau-nernst.github.io/fa-5090/ldmatrix.svg)
*图：mma.m16n8k16 操作中 ldmatrix tiles 的顺序*

![ldmatrix K/V 加载](https://gau-nernst.github.io/fa-5090/ldmatrix_kv.svg)
*图：加载 K 和 V 数据时使用转置的 ldmatrix*

**重要说明**：
> "只有 Q 在 MMA 中充当 A。K 和 V 都在各自的 MMA 中充当 B，尽管 K 需要转置的 `ldmatrix` 才能获得正确的布局。"

### MMA 操作和寄存器布局

kernel 围绕 **`mma.m16n8k16`** 指令构建计算：

#### 寄存器分配（C++ 数组声明）

```cpp
// MMA 常量
const int MMA_M = 16, MMA_N = 8, MMA_K = 16;

// 每个线程的寄存器数组
uint32_t Q_rmem[WARP_Q/16][DIM/16][4];          // Query (乘数 A)
uint32_t K_rmem[BLOCK_KV/8][DIM/16][2];         // Key (乘数 B)
uint32_t V_rmem[BLOCK_KV/16][DIM/8][2];         // Value (乘数 B)
float O_rmem[WARP_Q/16][DIM/8][4];              // 输出累加器

// 中间结果
float S_rmem[WARP_Q/16][BLOCK_KV/8][4];         // 注意力分数
uint32_t P_rmem[WARP_Q/16][BLOCK_KV/16][4];     // 注意力概率
```

**配置示例**（WARP_Q=32, DIM=128, BLOCK_KV=64）：
```
Q_rmem: [32/16][128/16][4] = [2][8][4] = 64 个 uint32_t = 64 寄存器
K_rmem: [64/8][128/16][2] = [8][8][2] = 128 寄存器
V_rmem: [64/16][128/8][2] = [4][16][2] = 128 寄存器
O_rmem: [2][16][4] = 128 个 float = 128 寄存器
S_rmem: [2][8][4] = 64 寄存器
```

**Warp 分配**：
> "threadblock 中的每个 warp 处理 `tile_Q` 的一部分，沿着 Q 序列长度维度进行分割。"

### 在线 Softmax 实现

维护每行在 KV 迭代中的状态：
- 行最大值（`m`）
- 未归一化的输出（`tile_O`）
- 归一化器和（`sumexp`）

#### 算法步骤

1. **计算行最大值**：
   - 使用 4 线程组内的蝶式归约（butterfly reduction）
   - 通过 `__shfl_xor_sync()` 实现

![行归约操作](https://gau-nernst.github.io/fa-5090/row_reduction.svg)
*图：MMA 输出上的行归约操作*

![蝶式归约](https://gau-nernst.github.io/fa-5090/butterfly_reduction.svg)
*图：4 线程内使用 __shfl_xor_sync() 的蝶式归约*

2. **最大值减法和指数运算**：
   - 逐元素应用：`exp(score - max)`

3. **重新缩放之前的输出**：
   - 乘以缩放因子：`exp(old_m - new_m)`

4. **数据打包**：
   - 将归一化的 logits 转换为 BF16 寄存器
   - 用于第二个 MMA，无需线程间数据移动
   - 利用 MMA 输出和乘数 A 之间相同的寄存器布局

### 性能特征

**基准测试配置**：
```
batch_size=1
num_heads=8
len_query=4096
len_kv=8192
head_dim=128
```

**Profiling 发现的瓶颈**：
- Shared Memory 的 bank conflicts
- 内存停顿（memory stalls）

![版本 1 Warp 状态](https://gau-nernst.github.io/fa-5090/v1_warp_state.png)
*图：版本 1 的 Nsight Compute Warp 状态统计*

---

## 版本 2：Shared Memory Swizzling

### 性能提升

- **吞吐量**：142.87 → **181.11 TFLOPS**（+27%）
- **SOL 效率**：68.20% → **86.45%**

### 问题识别

在 `ldmatrix` 操作中出现 **bank conflicts**：
- GPU 有 32 个 shared memory banks
- 从 64 元素宽的行加载 8×8 tiles 时造成 **"8-way bank conflicts"**

![Bank Conflicts 可视化](https://gau-nernst.github.io/fa-5090/bank_conflicts.svg)
*图：8×64 BF16 tile 的 memory bank 分布（展示 bank conflict）*

![ldmatrix Bank Conflicts](https://gau-nernst.github.io/fa-5090/ldmatrix_bank_conflicts.png)
*图：Nsight Compute 显示的实际 vs 理想 L1 Wavefronts Shared 指标*

**原因**：
> "当我们移动到下一行时，我们再次访问相同的 memory banks"

### 解决方案：XOR-based Swizzling

应用基于 XOR 的地址转换，在 banks 之间排列数据。

#### Swizzling 公式

```cpp
swizzled_index = index ^ (bits_to_xor << 4)
```

其中：
```cpp
row_idx = (index / STRIDE) % 8
bits_to_xor = row_idx / max(64 / STRIDE, 1)
```

**参数说明**：
- **index**：原始的 shared memory 偏移量（字节）
- **STRIDE**：行步长（字节），即 tile 宽度
- **row_idx**：在 8 行 swizzling 组内的行索引（0-7）
- **bits_to_xor**：从行索引和步长派生的 XOR 位

**为什么左移 4 位？**
- 位 0-3 由于 16 字节对齐约束总是为零
- 位 4-6 决定 bank 索引
- 左移 4 位使 XOR 操作修改位 4-6，控制 memory bank 选择，而不影响低位对齐位

#### 关键洞察

> "XOR 操作很好地排列数据"，因为它在输入和输出之间具有一对一映射。

行索引位可以与决定 bank 的地址位进行 XOR，因为：
- 移动到下一行会增加特定的地址位
- 同时保持其他位不变

### Swizzling 效果

**之前**：8-way bank conflict
```
Row 0: threads → bank 0, 1, 2, ..., 7, 0, 1, ...
Row 1: threads → bank 0, 1, 2, ..., 7, 0, 1, ...  ← 冲突！
```

**之后**：无 bank conflict
```
Row 0: threads → bank 0, 1, 2, 3, 4, 5, 6, 7
Row 1: threads → bank 4, 5, 6, 7, 0, 1, 2, 3  ← 交错分布
```

![版本 2 Warp 状态](https://gau-nernst.github.io/fa-5090/v2_warp_state.png)
*图：版本 2 的 Nsight Compute Warp 状态统计（bank conflicts 已消除）*

---

## 版本 3：2 阶段流水线

### 性能提升

- **吞吐量**：181.11 → **189.84 TFLOPS**（+4.9%）
- **SOL 效率**：86.45% → **90.62%**

### 解决的问题

**长延迟的 global memory 操作**使 Tensor Cores 在等待数据传输时停顿。

### 流水线设计

维护两个正在进行的阶段，使用：
- `cp.async.commit_group`
- `cp.async.wait_group`

**核心思想**：
> "为迭代 N+1 预取数据，同时在迭代 N 的数据上进行计算"

#### 流水线阶段

```
时间 t:   处理 KV_tile[i]     预取 KV_tile[i+1]
时间 t+1: 处理 KV_tile[i+1]   预取 KV_tile[i+2]
```

### 实现细节

> "在第 1 个 MMA 后放置 `load_V()` 可使 Tensor Cores 立即保持生产力；在在线 softmax 之前定位 `load_K()` 可在 CUDA cores 处理 softmax 计算时保持内存引擎利用率。"

#### K-V-K 预取模式

```cpp
// 伪代码
while (kv_iter < num_kv_tiles) {
    load_K_async(kv_iter + 1);        // 预取下一个 K

    mma_QK(kv_iter);                  // 计算当前 S = Q @ K^T

    load_V_async(kv_iter + 1);        // 预取下一个 V

    online_softmax(kv_iter);          // Softmax 计算

    mma_PV(kv_iter);                  // 计算当前 O += P @ V

    kv_iter++;
}
```

### 性能权衡与配置调整

**重要变化**：
> "我不得不将 `BLOCK_KV` 从 64 减少到 32，因为我们现在为 K 和 V 使用了两个缓冲区"

尽管 `BLOCK_KV` 减少，但通过重叠计算和内存传输，仍实现了 4.9% 的提升。这说明流水线带来的延迟隐藏效益超过了 tile 大小减小的负面影响。

![版本 3 Warp 状态](https://gau-nernst.github.io/fa-5090/v3_warp_state.png)
*图：版本 3 的 Nsight Compute Warp 状态统计（长延迟已改善）*

---

## 版本 4：ldmatrix.x4 优化

### 性能提升

- **吞吐量**：189.84 → **194.33 TFLOPS**（+2.4%）
- **SOL 效率**：90.62% → **92.76%**

### 优化方法

将两个 `ldmatrix.x2` 指令替换为单个 `ldmatrix.x4` 调用，用于加载 key 和 value 矩阵。

**配置说明**：
- `BLOCK_KV` 维持在 32（与版本 3 相同）
- 主要优化在于指令级别

#### 优化前

```cpp
// 加载 8×16 的 K tile（需要 2 次调用）
ldmatrix.x2  // 加载前 8 个元素
ldmatrix.x2  // 加载后 8 个元素
```

#### 优化后

```cpp
// 加载完整的 K tile（1 次调用）
ldmatrix.x4  // 一次加载全部 32 个元素
```

### 指令级优化

**减少指令发射开销**：
- 使用更大的 tiles（n8k32 或 n16k16 而非 n8k16）
- 需要不同的循环计数器递增方式
- 通过指令级优化实现"非平凡的改进"

### 权衡分析

虽然改变了 tile 大小，但减少的指令数量带来了显著的性能提升。

![ldmatrix.x4 选项](https://gau-nernst.github.io/fa-5090/ldmatrix_x4_B.svg)
*图：使用 ldmatrix.x4 加载乘数 B 的可能选项*

---

## 版本 5：优化的流水线

### 性能提升

- **吞吐量**：194.33 → **197.74 TFLOPS**（+1.8%）
- **SOL 效率**：92.76% → **94.39%**

### 核心改进

**消除 V 的冗余双缓冲**，通过战略性同步点使用单个 shared buffer。

#### 关键洞察

- **K 需要双缓冲**：因为 K 的预取发生在第 2 个 MMA 期间
- **V 只需单缓冲**：因为 V 的预取发生在第 1 个 MMA 期间

#### 优化的预取模式

```cpp
// 版本 4（V 需要双缓冲）
buffer_V[0], buffer_V[1]  // 两个 V buffers
buffer_K[0], buffer_K[1]  // 两个 K buffers

// 版本 5（V 只需单缓冲）
buffer_V                  // 一个 V buffer
buffer_K[0], buffer_K[1]  // 两个 K buffers
```

### 资源效率提升

释放的 shared memory 使得可以：
- 将 `BLOCK_KV` 从 32 增加回 64
- 接近实际性能上限（约 94.39% SOL）

**作者说明**：
> "更高效地使用 shared memory 意味着我们可以增加一些 tile 大小。我将 `BLOCK_KV` 从 32 增加回 64"

#### 同步策略

```cpp
// 在第 1 个 MMA 期间预取 V（安全）
mma_QK();
load_V_async();  // V 可以立即预取

// 在第 2 个 MMA 之前同步 V
sync_V();
mma_PV();        // 使用新的 V 数据
```

---

## 性能对比与分析

### 完整基准测试对比表

| Kernel | TFLOPS | % SOL |
|--------|--------|-------|
| **F.sdpa() (CuDNN)** | **203.61** | **97.19%** |
| flash-attn | 190.58 | 90.97% |
| **v5 (更好的流水线)** | **197.74** | **94.39%** |
| v4 (K 和 V 的 ldmatrix.x4) | 194.33 | 92.76% |
| v3 (2 阶段流水线) | 189.84 | 90.62% |
| v2 (shared memory swizzling) | 181.11 | 86.45% |
| F.sdpa() (Flash Attention) | 186.73 | 89.13% |
| v1 (基础版本) | 142.87 | 68.20% |

**硬件环境**：
- GPU：RTX 5090（400W）
- 理论峰值：**209.5 TFLOPS**（BF16 操作）

### Profiling 洞察

#### 版本 1 的瓶颈

初始实现揭示了两个主要性能抑制因素：

1. **Stall Math Pipe Throttle**：warps 忙于 tensor core 操作
2. **Stall Short Scoreboard**（主要瓶颈）：表示 shared memory 访问延迟

**内存分析**：
- 暴露了 "Shared Load Bank Conflicts"
- 连续线程在 `ldmatrix` 操作期间访问相同的 memory banks

#### 渐进式优化影响

**版本 2（Swizzling）**：
- 通过基于 XOR 的 shared memory 地址排列解决 bank conflicts
- 消除了 short scoreboard stall
- 实现 8-way conflict 减少
- 性能提升：**26%**

**版本 3（2 阶段流水线）**：
- 通过 `cp.async` groups 引入异步 global-to-shared 传输
- 重叠数据移动与计算
- Profiling 数据显示 "Stall Long Scoreboard" 已解决
- 性能提升：**4.3%**（尽管 `BLOCK_KV` 从 64 减少到 32）

**版本 4（指令减少）**：
- 用单个 `ldmatrix.x4` 操作替换双 `ldmatrix.x2` 调用
- 减少指令调度开销
- 性能提升：**2.3%**

**版本 5（高效缓冲）**：
- 识别出 K 需要双缓冲而 V 需要单缓冲
- 允许重新分配 shared memory 资源
- `BLOCK_KV` 恢复为 64
- 性能提升：**1.7%**

### 理论性能讨论

#### Shared Memory 限制

作者指出：
> "消费级 GPU 与服务器对应产品相比，通常具有适度的 shared memory 大小"

这限制了架构选择。

#### 与 CuDNN 的性能差距

v5（94.39% SOL）与 CuDNN（97.19% SOL）之间的差距代表约 **3% 的性能开销**，可能归因于：
- 指令调度复杂性
- 极端利用率水平的内存带宽饱和

#### CuDNN 的缓冲策略

CuDNN 的实现使用约 **2.5 个缓冲区分配**——这种分数模式表明存在尚未探索的替代架构方法。

### 性能进化可视化

```
版本进化：
v1: ████████████████████████████████████          68.20%
v2: ████████████████████████████████████████████  86.45% (+18.25%)
v3: ██████████████████████████████████████████████ 90.62% (+4.17%)
v4: ███████████████████████████████████████████████ 92.76% (+2.14%)
v5: ████████████████████████████████████████████████ 94.39% (+1.63%)
CuDNN: ████████████████████████████████████████████████ 97.19%
```

---

## 总结与未来方向

### 关键成就

1. **从零开始实现 Flash Attention**，达到理论峰值的 **94.39%**
2. **系统性优化过程**，每个版本都有明确的性能瓶颈和解决方案
3. **接近 CuDNN 性能**（97.19%），仅相差 2.8 个百分点

### 技术要点总结

| 优化技术 | 性能影响 | 关键洞察 |
|---------|---------|---------|
| **Shared Memory Swizzling** | +26% | 通过 XOR 消除 bank conflicts |
| **2 阶段流水线** | +4.9% | 重叠计算与内存传输 |
| **ldmatrix.x4** | +2.4% | 减少指令发射开销 |
| **优化的流水线** | +1.8% | V 单缓冲，K 双缓冲 |

### 学到的经验

1. **Bank conflicts 是主要瓶颈**：在优化的早期阶段最为显著
2. **流水线至关重要**：隐藏 global memory 延迟
3. **指令级优化**：在接近极限时很重要
4. **内存层次结构**：理解何时需要双缓冲与单缓冲

### 未来研究方向

作者确定了几个发展方向：

1. **反向传播实现**：完整的训练 kernel
2. **量化 attention kernels**：
   - 特别是 5090 上的 **NVFP4**
   - MXFP8 支持
3. **基于 TMA 的 warp 专门化设计**：
   - 利用 Hopper/Blackwell 的新特性
4. **PagedAttention 集成**：
   - 用于生产服务系统
   - 内存效率优化

### 开源贡献

完整的实现代码开源在 GitHub，包括：
- 所有 5 个版本的完整源代码
- 基准测试脚本
- Profiling 工具和分析

---

## 技术细节补充

### MMA 指令详解

#### mma.m16n8k16 指令

```
输入：
- A (Q): 16×16 矩阵，4 个寄存器/线程
- B (K 或 V): 8×16 矩阵，2 个寄存器/线程

输出：
- C: 16×8 矩阵，4 个寄存器/线程
```

**线程组织**：
- 32 个线程（1 个 warp）
- 每个线程处理 2×2 元素分组

### Shared Memory 布局

```cpp
// K 的 Shared Memory 布局（BLOCK_KV × DIM）
[64 × 128] 元素
stride = 128（行主序）

// 使用 swizzling 后
effective_stride = swizzled(128)  // 避免 bank conflicts
```

### 寄存器使用分析

#### 每个 Warp 的寄存器消耗

```
Q:     [WARP_Q/16][DIM/16][4] = [128/16][128/16][4] = 256 寄存器
K:     [BLOCK_KV/8][DIM/16][2] = [64/8][128/16][2] = 128 寄存器
V:     同 K = 128 寄存器
S:     [WARP_Q/16][BLOCK_KV/8][4] = [8][8][4] = 256 寄存器
O:     [WARP_Q/16][DIM/8][4] = [8][16][4] = 512 寄存器

总计：~1280 寄存器/warp
```

### 在线 Softmax 的数学原理

#### 关联性更新公式

给定当前状态 `(m_old, sumexp_old, O_old)` 和新 tile 的分数 `S_new`：

```python
# 步骤 1：计算新的最大值
m_new = max(m_old, max(S_new))

# 步骤 2：重新缩放因子
scale_old = exp(m_old - m_new)
scale_new = exp(max(S_new) - m_new)

# 步骤 3：更新累加器
O_new = scale_old * O_old + scale_new * (softmax_local(S_new) @ V_new)
sumexp_new = scale_old * sumexp_old + scale_new * sum(exp(S_new - m_new))

# 步骤 4：最终归一化（在所有 tiles 处理完后）
O_final = O_new / sumexp_new
```

### Swizzling 的位操作详解

```cpp
// 示例：64 个元素宽的行，BF16 类型（2 字节）
// 行步长（STRIDE）：128 字节
// Bank 宽度：4 字节 → 每个 bank 2 个 BF16

原始地址映射: index = row * 128 + col * 2 (字节)
Bank 索引: (index / 4) % 32

// 对于第 0 行、第 0 列：
index = 0 * 128 + 0 * 2 = 0
bank = 0 / 4 % 32 = 0

// 对于第 1 行、第 0 列（未 swizzle）：
index = 1 * 128 + 0 * 2 = 128
bank = 128 / 4 % 32 = 0  ← 冲突！

// 使用 swizzling 计算：
row_idx = (128 / 128) % 8 = 1
bits_to_xor = 1 / max(64/128, 1) = 1 / 1 = 1
swizzled = 128 ^ (1 << 4) = 128 ^ 16 = 144
bank = 144 / 4 % 32 = 4  ← 不同的 bank，无冲突！

// 对于第 2 行、第 0 列：
index = 2 * 128 = 256
row_idx = 2, bits_to_xor = 2
swizzled = 256 ^ (2 << 4) = 256 ^ 32 = 288
bank = 288 / 4 % 32 = 8  ← 又一个不同的 bank！
```

**效果**：通过 XOR 操作，连续行访问不同的 banks，消除了 8-way bank conflict。

---

## 附录：性能测试配置

### 硬件配置

```
GPU: NVIDIA GeForce RTX 5090
功耗: 400W
计算能力: sm_120 (Blackwell)
理论峰值 (BF16): 209.5 TFLOPS
Shared Memory: ~100 KB/SM（消费级 GPU）
```

### 软件配置

```
CUDA: 12.x
编译器: nvcc
优化标志: -O3, -use_fast_math
精度: BF16 (Brain Float 16)
```

### 基准测试参数

```python
batch_size = 1
num_heads = 8
len_query = 4096   # Q 序列长度
len_kv = 8192      # KV 序列长度
head_dim = 128     # 注意力头维度

# Kernel 配置
BLOCK_Q = 128
BLOCK_KV = 64      # v5 版本
NUM_WARPS = 4
```

### 性能测量方法

使用 NVIDIA Nsight Compute 进行 profiling：
- **FLOPS 计算**：基于 MMA 操作数和 kernel 执行时间
- **SOL (Speed of Light)**: `实际 TFLOPS / 理论峰值 TFLOPS × 100%`
- **瓶颈分析**：Stall reasons, memory throughput, instruction mix

---

## 参考资源

1. **原文链接**：https://gau-nernst.github.io/fa-5090/
2. **Flash Attention 论文**：
   - Flash Attention: Fast and Memory-Efficient Exact Attention with IO-Awareness
   - Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning
3. **GPU-MODE 讲座**：https://github.com/gpu-mode/lectures
4. **相关博客**：
   - 快速 matmul kernel 编写教程
   - CUDA Tensor Core 编程指南
5. **开源实现**：作者的 GitHub 仓库

---

## 致谢

感谢原作者 **gau-nernst** 的详细技术分享，为社区提供了宝贵的学习资源。

---

**文档信息**：
- 翻译日期：2026-01
- 译者：根据原始博客翻译
- 版本：v1.0

**免责声明**：本文档为技术翻译，如有理解偏差，请以原文为准。
