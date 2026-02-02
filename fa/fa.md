# Flash Attention 详解

本文档详细介绍 Flash Attention 算法的原理、演进和实现。

---

## 目录

- [Flash Attention 版本对比](#flash-attention-版本对比)
  - [Flash Attention v1 (2022)](#flash-attention-v1-2022)
  - [Flash Attention v2 (2023)](#flash-attention-v2-2023)
  - [Flash Attention v3 (2024)](#flash-attention-v3-2024)
  - [版本对比总结](#版本对比总结)
- [代码示例](#代码示例)
  - [GM2SM (Global Memory to Shared Memory)](#gm2sm)

---

## Flash Attention 版本对比

### Flash Attention v1 (2022)

**论文**: *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness and Recomputation*
- 作者: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford)
- 发布: 2022年5月

#### 核心创新

1. **IO-Aware 算法设计**
   - 从 GPU 内存层次结构出发设计算法
   - 识别到标准 Attention 的主要瓶颈是 HBM (High Bandwidth Memory) 访问
   - 设计目标: 最小化 HBM 读写次数

2. **Tiling 策略**
   ```
   标准 Attention: O(N²) HBM 访问
   Flash Attention v1: O(N² d² M⁻¹) HBM 访问

   其中:
   - N: 序列长度
   - d: head dimension
   - M: SRAM 大小 (Shared Memory)
   ```

3. **Online Softmax 算法**
   - 无需存储完整的 attention matrix
   - 分块计算 softmax，使用增量更新
   - 数学基础:
     ```
     softmax(x) = exp(x) / sum(exp(x))

     分块更新:
     m_new = max(m_old, m_block)
     l_new = exp(m_old - m_new) * l_old + exp(m_block - m_new) * l_block
     ```

4. **Recomputation in Backward Pass**
   - Forward pass: 不保存 attention matrix (节省 O(N²) 内存)
   - Backward pass: 从 Q, K 重新计算 attention
   - 权衡: 用计算换内存

#### 性能提升

- **速度**: 2-4x faster than standard attention
- **内存**: 从 O(N²) 降至 O(N)
- **精度**: Exact attention (无近似，数值等价)
- **序列长度**: 支持 64K tokens (原 2K → 64K)

#### 局限性

1. 并行化策略不够优化
2. Non-matmul FLOPs 占比较高
3. 未充分利用现代 GPU 硬件特性
4. 对不同 head dimension 的适应性不足

---

### Flash Attention v2 (2023)

**论文**: *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*
- 作者: Tri Dao (Princeton)
- 发布: 2023年7月

#### 主要改进

1. **改进的并行化策略**

   **v1 的并行化**:
   ```
   外层: 并行化 batch 和 num_heads
   内层: 对每个 head，顺序处理 Q 的 blocks
   ```

   **v2 的并行化**:
   ```
   外层: 并行化所有维度 (batch, num_heads, q_block_idx)
   结果: 更细粒度的并行，更好的 GPU 利用率
   ```

2. **减少 Non-Matmul FLOPs**

   **优化前 (v1)**:
   - Attention 计算中 non-matmul 操作占 30-40%
   - 包括: rescaling, masking, softmax, dropout

   **优化后 (v2)**:
   - 将多个操作融合到一个 kernel
   - Non-matmul FLOPs 降至 15-20%
   - 更好地利用 Tensor Cores

3. **优化的 Work Partitioning**

   **v1 问题**:
   - 不同 warp 的工作量不均衡
   - 某些 warp 空闲等待

   **v2 改进**:
   - 在 Q 的 sequence length 维度分块
   - 每个 thread block 处理更小的 Q block
   - 在 K 的 sequence length 维度并行
   - 减少同步开销

4. **多查询注意力 (MQA/GQA) 支持**
   - Multi-Query Attention (MQA): 多个 Q 共享 K, V
   - Grouped-Query Attention (GQA): Q 分组共享 K, V
   - 优化内存访问模式

#### 算法改进

**v1 伪代码**:
```python
for q_block in Q_blocks:
    for kv_block in KV_blocks:
        # Load Q, K, V blocks
        # Compute attention
        # Update output incrementally
```

**v2 伪代码**:
```python
# 所有 q_blocks 并行
parallel_for q_block in Q_blocks:
    for kv_block in KV_blocks:
        # 优化的内存访问
        # 融合的操作
        # 减少同步
```

#### 性能提升

与 Flash Attention v1 对比:
- **A100 GPU**:
  - 短序列 (512): ~1.3x faster
  - 长序列 (2K): ~1.8x faster
  - 超长序列 (8K): ~2x faster
- **Memory**: 保持 O(N) 复杂度
- **Throughput**: 接近硬件理论峰值 (达到 70-80% 的 FLOPS)

与标准 Attention 对比:
- **速度**: 5-9x faster
- **内存**: 减少 10-20x

#### 适用场景

- 长序列训练 (GPT, LLaMA)
- 多模态模型 (CLIP, Flamingo)
- 长文本理解
- 代码生成模型

---

### Flash Attention v3 (2024)

**论文**: *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*
- 作者: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
- 发布: 2024年3月
- **针对 Hopper 架构 (H100) 优化**

#### 核心创新

1. **利用 Hopper 新硬件特性**

   **Tensor Memory Accelerator (TMA)**:
   ```
   作用: 异步数据传输
   - CPU 发起, GPU 自动完成数据搬运
   - 释放 threads 做其他计算
   - 降低延迟，提高吞吐
   ```

   **Warpgroup & Warp Specialization**:
   ```
   Warpgroup: 4个warp (128 threads) 作为一组
   - 专门化: 不同 warpgroup 执行不同任务
   - Producer warpgroup: 加载数据
   - Consumer warpgroup: 执行计算
   - Overlap: 数据传输和计算重叠
   ```

   **Thread Block Clusters**:
   ```
   多个 Thread Blocks 组成 Cluster
   - 跨 SM 协作
   - 共享 L2 Cache
   - 更灵活的并行策略
   ```

2. **异步流水线设计**

   **v2 同步流程**:
   ```
   Load Q → Wait → Load K,V → Wait → Compute → Wait → Store
   串行执行，GPU 利用率低
   ```

   **v3 异步流程**:
   ```
   Producer Warpgroup:  Load Q → Load K,V → Load next...
                           ↓         ↓
   Consumer Warpgroup:          Compute → Compute → ...

   并行执行，隐藏延迟
   ```

3. **低精度计算 (FP8)**

   **FP8 E4M3 格式**:
   ```
   1 bit sign | 4 bits exponent | 3 bits mantissa

   优势:
   - 2x 吞吐量 (相比 FP16/BF16)
   - H100 支持 FP8 Tensor Core
   - 动态范围足够 Attention 计算
   ```

   **Incoherent Processing**:
   ```
   Q, K 使用不同的 scaling
   - Per-block rescaling
   - 保持数值稳定性
   - 精度损失 < 0.1%
   ```

4. **Softmax 的硬件级优化**

   **利用 Hopper WGMMA (Warpgroup MMA)**:
   ```
   传统: Matmul → Rescale → Softmax → Matmul
   v3: 融合操作，减少数据搬运
   ```

5. **新的调度策略**

   **Ping-pong Scheduling**:
   ```
   Double buffering in Shared Memory
   - Buffer A: 正在计算
   - Buffer B: 正在加载下一批数据
   - 无等待切换
   ```

#### 算法流程

**Flash Attention v3 伪代码**:
```python
# 初始化 warpgroup 角色
producer_wg = warpgroup(0, 1)  # 前2个warpgroup负责加载
consumer_wg = warpgroup(2, 3)  # 后2个warpgroup负责计算

# Producer warpgroup
if is_producer:
    for kv_block in KV_blocks:
        # 使用 TMA 异步加载
        tma_load_async(Q_smem, Q_gmem, ...)
        tma_load_async(K_smem, K_gmem, ...)
        tma_load_async(V_smem, V_gmem, ...)
        # 不等待，继续下一轮

# Consumer warpgroup
if is_consumer:
    for kv_block in KV_blocks:
        # 等待数据就绪
        wait_group(kv_block)

        # FP8 计算
        S = matmul_fp8(Q_fp8, K_fp8^T)

        # Online softmax with FP8
        P = softmax_fp8(S, scale)

        # 累积输出
        O = matmul_fp8(P_fp8, V_fp8)
```

#### 性能提升

**与 Flash Attention v2 对比** (H100):

| 场景 | v2 TFLOPS | v3 TFLOPS | 提升 |
|-----|-----------|-----------|------|
| Seq=512, Head=64 | 450 | 740 | 1.64x |
| Seq=2K, Head=64 | 580 | 950 | 1.64x |
| Seq=8K, Head=64 | 620 | 989 | 1.59x |
| Seq=16K, Head=128 | 650 | 985 | 1.51x |

- **吞吐量**: 接近 H100 FP8 理论峰值 (990 TFLOPS)
- **效率**: 90-95% 的硬件利用率
- **精度**: FP8 精度损失 < 0.1% perplexity

**实际应用性能** (GPT-3 175B 训练):
- 端到端加速: 1.3-1.5x (相比 v2)
- 内存节省: 与 v2 相同 (O(N))

#### 局限性

1. **硬件依赖**: 仅支持 Hopper (H100, H200) 及更新架构
2. **FP8 限制**: 某些任务对精度敏感，需要评估
3. **复杂性**: 实现难度大，调试困难
4. **兼容性**: 不向后兼容旧 GPU

---

### 版本对比总结

#### 功能对比表

| 特性 | Flash Attention v1 | Flash Attention v2 | Flash Attention v3 |
|-----|-------------------|-------------------|-------------------|
| **发布时间** | 2022年5月 | 2023年7月 | 2024年3月 |
| **目标架构** | Ampere (A100) | Ampere/Ada | Hopper (H100) |
| **核心优化** | IO-aware, Tiling | 并行化, Work Partitioning | 异步执行, FP8 |
| **内存复杂度** | O(N) | O(N) | O(N) |
| **计算精度** | FP16/BF16 | FP16/BF16 | FP8/FP16/BF16 |
| **并行策略** | Batch + Heads | Batch + Heads + Q blocks | Warpgroup + Clusters |
| **异步执行** | ❌ | ❌ | ✅ (TMA) |
| **MQA/GQA 支持** | ❌ | ✅ | ✅ |
| **相比标准 Attention 加速** | 2-4x | 5-9x | 7-12x (H100) |
| **相比前一版本加速** | - | 1.5-2x | 1.5-1.7x |

#### 速度对比 (A100, Seq=2K, Head=64)

```
标准 Attention:     100 TFLOPS (baseline)
Flash Attention v1: 250 TFLOPS (2.5x)
Flash Attention v2: 450 TFLOPS (4.5x)
Flash Attention v3: N/A (需要 H100)
```

#### 速度对比 (H100, Seq=2K, Head=64)

```
标准 Attention:     120 TFLOPS (baseline)
Flash Attention v2: 580 TFLOPS (4.8x)
Flash Attention v3: 950 TFLOPS (7.9x)
```

#### 适用场景建议

**Flash Attention v1**:
- 遗留系统，需要兼容性
- 研究理解 IO-aware 算法
- 不推荐生产使用

**Flash Attention v2**:
- **生产环境首选** (A100, A6000, RTX 40 系列)
- 长序列训练 (Seq > 2K)
- 最广泛的硬件支持
- 稳定性最好

**Flash Attention v3**:
- **H100/H200 最佳选择**
- 极限性能追求
- 大规模模型训练 (GPT-4 级别)
- FP8 推理加速
- 需要评估精度影响

---

## 关键技术深入

### 1. Online Softmax 算法

所有版本的核心技术，允许分块计算 softmax。

#### 数学原理

标准 Softmax:
```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

分块更新公式:
```
输入: 已有 m_old (max), l_old (sum), O_old (output)
新块: x_new

1. 计算新块的 max 和 sum:
   m_new_block = max(x_new)
   l_new_block = Σ exp(x_new - m_new_block)

2. 更新全局 max:
   m_new = max(m_old, m_new_block)

3. 重新缩放并更新 sum:
   l_new = exp(m_old - m_new) * l_old + exp(m_new_block - m_new) * l_new_block

4. 重新缩放并更新 output:
   O_new = diag(exp(m_old - m_new)) * O_old + exp(m_new_block - m_new) * softmax(x_new) @ V_new
```

#### 实现伪代码

```python
def online_softmax_attention(Q, K, V, block_size):
    Tr = (seq_len + block_size - 1) // block_size
    Tc = (seq_len + block_size - 1) // block_size

    # 初始化
    O = zeros_like(Q)
    m = -inf * ones(Tr)  # 每个 Q block 的 max
    l = zeros(Tr)        # 每个 Q block 的 sum

    # 外层循环: Q blocks
    for i in range(Tr):
        Q_i = Q[i*block_size:(i+1)*block_size]

        # 内层循环: K, V blocks
        for j in range(Tc):
            K_j = K[j*block_size:(j+1)*block_size]
            V_j = V[j*block_size:(j+1)*block_size]

            # 计算 attention scores
            S_ij = Q_i @ K_j.T  # [block_size, block_size]

            # 当前 block 的统计量
            m_ij = max(S_ij)
            P_ij = exp(S_ij - m_ij)
            l_ij = sum(P_ij)

            # 更新全局统计量
            m_new = max(m[i], m_ij)
            l_new = exp(m[i] - m_new) * l[i] + exp(m_ij - m_new) * l_ij

            # 更新 output (增量更新)
            O[i] = (l[i] * exp(m[i] - m_new) * O[i] +
                    exp(m_ij - m_new) * P_ij @ V_j) / l_new

            # 保存新的统计量
            m[i] = m_new
            l[i] = l_new

    return O
```

### 2. Backward Pass 优化

#### Flash Attention v1/v2 策略

```python
def backward_flash_attention(Q, K, V, dO):
    # Forward pass 时只保存 (O, m, l)，不保存 S, P
    # Backward 时重新计算

    dQ = zeros_like(Q)
    dK = zeros_like(K)
    dV = zeros_like(V)

    for i in range(Tr):
        Q_i = Q[i*block_size:(i+1)*block_size]
        dO_i = dO[i*block_size:(i+1)*block_size]

        for j in range(Tc):
            K_j = K[j*block_size:(j+1)*block_size]
            V_j = V[j*block_size:(j+1)*block_size]

            # 重新计算 forward pass (不需要额外内存)
            S_ij = Q_i @ K_j.T
            P_ij = softmax(S_ij)

            # 计算梯度
            dV_j += P_ij.T @ dO_i
            dP_ij = dO_i @ V_j.T
            dS_ij = softmax_backward(P_ij, dP_ij)
            dQ_i += dS_ij @ K_j
            dK_j += dS_ij.T @ Q_i

    return dQ, dK, dV
```

### 3. FP8 量化策略 (v3)

#### E4M3 vs E5M2

```
E4M3 (用于 Q, K, O):
- 1 bit sign, 4 bits exponent, 3 bits mantissa
- Dynamic range: 2^-6 to 2^8 (适合激活值)

E5M2 (用于梯度):
- 1 bit sign, 5 bits exponent, 2 bits mantissa
- Dynamic range: 2^-14 to 2^15 (适合梯度)
```

#### 量化方法

```python
def quantize_fp8(x, scale):
    # Per-block scaling
    block_max = max(abs(x))
    scale = block_max / FP8_MAX

    x_scaled = x / scale
    x_fp8 = cast_to_fp8(x_scaled)

    return x_fp8, scale

def dequantize_fp8(x_fp8, scale):
    x_fp32 = cast_to_fp32(x_fp8)
    return x_fp32 * scale
```

---

## 实现细节

### Shared Memory 布局

```cuda
// Flash Attention v2/v3 Shared Memory Layout
__shared__ float Q_smem[BLOCK_M][HEAD_DIM];    // Query block
__shared__ float K_smem[BLOCK_N][HEAD_DIM];    // Key block
__shared__ float V_smem[BLOCK_N][HEAD_DIM];    // Value block
__shared__ float O_smem[BLOCK_M][HEAD_DIM];    // Output accumulator

__shared__ float m_smem[BLOCK_M];  // Running max
__shared__ float l_smem[BLOCK_M];  // Running sum

// v3 额外的 FP8 buffer
__shared__ __nv_fp8_e4m3 Q_fp8[BLOCK_M][HEAD_DIM];
__shared__ __nv_fp8_e4m3 K_fp8[BLOCK_N][HEAD_DIM];
```

### Thread Block 配置

```cuda
// Flash Attention v2
dim3 block(WARP_SIZE, NUM_WARPS);  // 例如 (32, 4) = 128 threads
dim3 grid(num_q_blocks, num_heads, batch_size);

// Flash Attention v3 (Hopper)
dim3 block(WARP_SIZE, NUM_WARPGROUPS * WARPS_PER_WARPGROUP);
// 例如 (32, 2 * 4) = 256 threads
// 使用 Thread Block Cluster
cudaLaunchCooperativeKernel(kernel, grid, block, ...);
```

---

## 性能调优建议

### 1. 选择合适的 Block Size

```python
# 经验法则
BLOCK_M = BLOCK_N = {
    'A100': 128,      # 对应 Shared Memory 大小
    'H100': 256,      # Hopper 有更大的 Shared Memory
    'RTX 4090': 64,   # 消费级显卡
}

# 根据 head_dim 调整
if head_dim <= 64:
    BLOCK_M = BLOCK_N = 128
elif head_dim <= 128:
    BLOCK_M = BLOCK_N = 64
else:
    BLOCK_M = BLOCK_N = 32
```

### 2. Kernel Launch 配置

```python
# PyTorch 接口
from flash_attn import flash_attn_func

# 自动选择最优配置
output = flash_attn_func(
    q, k, v,
    dropout_p=0.0,
    softmax_scale=None,  # 默认 1/sqrt(head_dim)
    causal=True,         # Causal masking for autoregressive
    window_size=(-1, -1), # Local attention window
)

# 手动配置 (高级)
output = flash_attn_func(
    q, k, v,
    block_m=128,         # Q block size
    block_n=128,         # KV block size
    num_splits=1,        # Sequence parallelism
)
```

### 3. 内存优化

```python
# 启用 Gradient Checkpointing
from torch.utils.checkpoint import checkpoint

def attention_layer(q, k, v):
    return flash_attn_func(q, k, v, causal=True)

# 在 forward 时不保存中间结果
output = checkpoint(attention_layer, q, k, v)
```

---

## 代码示例

### GM2SM

Global Memory to Shared Memory 的异步传输实现。

```c++
#include <cuda_bf16.h>

template <int HEIGHT, int WIDTH, int TB_SIZE>
__device__ inline
void global_to_shared(uint32_t dst, const nv_bfloat16 *src, int src_stride, int tid) {
  constexpr int num_elems = 16 / sizeof(nv_bfloat16);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * TB_SIZE + tid) * num_elems;
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const uint32_t dst_addr = dst + (row * WIDTH + col) * sizeof(nv_bfloat16);
    const nv_bfloat16 *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
}
```

**代码说明**:
- `cp.async.cg.shared.global`: CUDA 异步复制指令 (Ampere+)
- `num_elems`: 每次传输 16 字节 (8 个 bfloat16)
- `src_stride`: 处理非连续内存布局
- 用于 Flash Attention v2 的数据加载阶段

### Flash Attention v2 Kernel 框架

```cuda
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_v2_kernel(
    const half* Q,  // [batch, num_heads, seq_len, head_dim]
    const half* K,
    const half* V,
    half* O,
    float* m,      // running max
    float* l,      // running sum
    int seq_len,
    int num_heads,
    float softmax_scale
) {
    // Thread block 配置
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared Memory
    __shared__ half Q_smem[BLOCK_M][HEAD_DIM];
    __shared__ half K_smem[BLOCK_N][HEAD_DIM];
    __shared__ half V_smem[BLOCK_N][HEAD_DIM];
    __shared__ half S_smem[BLOCK_M][BLOCK_N];  // Attention scores

    __shared__ float m_smem[BLOCK_M];  // 每行的 max
    __shared__ float l_smem[BLOCK_M];  // 每行的 sum

    // 加载 Q block 到 Shared Memory
    const int q_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM
                        + q_block_idx * BLOCK_M * HEAD_DIM;

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += blockDim.x / HEAD_DIM) {
        if (q_block_idx * BLOCK_M + i < seq_len) {
            // 使用 cp.async 加载
            global_to_shared(
                __cvta_generic_to_shared(&Q_smem[i][0]),
                &Q[q_offset + i * HEAD_DIM],
                HEAD_DIM,
                tid
            );
        }
    }

    // 初始化统计量
    if (tid < BLOCK_M) {
        m_smem[tid] = -INFINITY;
        l_smem[tid] = 0.0f;
    }
    __syncthreads();

    // 外层循环: 遍历所有 KV blocks
    const int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        // 加载 K, V blocks
        const int kv_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM
                             + kv_block_idx * BLOCK_N * HEAD_DIM;

        // Load K
        #pragma unroll
        for (int i = 0; i < BLOCK_N; i += blockDim.x / HEAD_DIM) {
            if (kv_block_idx * BLOCK_N + i < seq_len) {
                global_to_shared(
                    __cvta_generic_to_shared(&K_smem[i][0]),
                    &K[kv_offset + i * HEAD_DIM],
                    HEAD_DIM,
                    tid
                );
            }
        }

        // Load V (同样的模式)
        // ...

        __syncthreads();

        // 计算 S = Q @ K^T
        // 使用 WMMA (Warp Matrix Multiply-Accumulate)
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> Q_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> K_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> S_frag;

        wmma::fill_fragment(S_frag, 0.0f);

        #pragma unroll
        for (int k = 0; k < HEAD_DIM; k += 16) {
            wmma::load_matrix_sync(Q_frag, &Q_smem[warp_m][k], HEAD_DIM);
            wmma::load_matrix_sync(K_frag, &K_smem[warp_n][k], HEAD_DIM);
            wmma::mma_sync(S_frag, Q_frag, K_frag, S_frag);
        }

        // 应用 softmax_scale
        #pragma unroll
        for (int i = 0; i < S_frag.num_elements; i++) {
            S_frag.x[i] *= softmax_scale;
        }

        // Online Softmax Update
        float m_old = m_smem[row];
        float l_old = l_smem[row];

        // 计算当前 block 的 max
        float m_block = -INFINITY;
        #pragma unroll
        for (int i = 0; i < S_frag.num_elements; i++) {
            m_block = fmaxf(m_block, S_frag.x[i]);
        }
        m_block = warp_reduce_max(m_block);

        // 更新全局 max
        float m_new = fmaxf(m_old, m_block);

        // 计算 exp 和 sum
        float l_block = 0.0f;
        #pragma unroll
        for (int i = 0; i < S_frag.num_elements; i++) {
            float p = expf(S_frag.x[i] - m_new);
            S_frag.x[i] = p;
            l_block += p;
        }
        l_block = warp_reduce_sum(l_block);

        // 更新全局 sum
        float l_new = expf(m_old - m_new) * l_old + l_block;

        // 重新缩放并更新 output
        float scale_old = expf(m_old - m_new) * l_old / l_new;
        float scale_new = 1.0f / l_new;

        // O = scale_old * O + scale_new * (P @ V)
        // 使用 WMMA 计算 P @ V
        // ...

        // 保存新的统计量
        m_smem[row] = m_new;
        l_smem[row] = l_new;

        __syncthreads();
    }

    // 写回 output 到 Global Memory
    // ...
}
```

### Flash Attention v3 异步版本 (简化)

```cuda
// Hopper 特性: Warpgroup 和 TMA
template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void flash_attention_v3_kernel(
    const __nv_fp8_e4m3* Q,
    const __nv_fp8_e4m3* K,
    const __nv_fp8_e4m3* V,
    __nv_fp8_e4m3* O,
    float* m, float* l,
    int seq_len
) {
    // Warpgroup 配置
    const int warpgroup_id = threadIdx.x / 128;  // 4 warps per warpgroup
    const bool is_producer = (warpgroup_id < 2);  // 前2个warpgroup加载数据
    const bool is_consumer = (warpgroup_id >= 2); // 后2个warpgroup计算

    __shared__ __nv_fp8_e4m3 Q_smem[2][BLOCK_M][HEAD_DIM];  // Double buffering
    __shared__ __nv_fp8_e4m3 K_smem[2][BLOCK_N][HEAD_DIM];
    __shared__ __nv_fp8_e4m3 V_smem[2][BLOCK_N][HEAD_DIM];

    int buffer_idx = 0;

    // Producer warpgroup: 异步加载数据
    if (is_producer) {
        for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
            // 使用 TMA (Tensor Memory Accelerator) 异步加载
            if (warpgroup_id == 0) {
                // 加载 K
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared.global.tile"
                    " [%0], [%1], [%2];"
                    :: "r"(__cvta_generic_to_shared(&K_smem[buffer_idx][0][0])),
                       "l"(&K[kv_block_idx * BLOCK_N * HEAD_DIM]),
                       "r"(tma_descriptor_K)
                );
            } else {
                // 加载 V
                // 类似的 TMA 指令
            }

            buffer_idx ^= 1;  // 切换 buffer
        }
    }

    // Consumer warpgroup: 执行计算
    if (is_consumer) {
        for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
            // 等待数据就绪 (Producer 已加载)
            asm volatile("cp.async.wait_group %0;" :: "n"(0));

            // 使用 Warpgroup MMA (WGMMA) 执行 FP8 矩阵乘法
            asm volatile(
                "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3 "
                "{%0, %1, %2, %3}, "
                "{%4, %5}, "
                "{%6, %7};"
                : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1])
            );

            // Online softmax (FP32 精度)
            // ...

            buffer_idx ^= 1;
        }
    }

    __syncthreads();
}
```

---

## 参考文献

### 论文

1. **Flash Attention v1**
   - [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness and Recomputation](https://arxiv.org/abs/2205.14135)
   - Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
   - NeurIPS 2022

2. **Flash Attention v2**
   - [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
   - Tri Dao
   - ICLR 2024

3. **Flash Attention v3**
   - [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
   - Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
   - 2024

### 官方实现

- **GitHub**: https://github.com/Dao-AILab/flash-attention
- **PyTorch 集成**: `pip install flash-attn`
- **HuggingFace Transformers**: 内置支持

### 相关技术

- **Online Softmax**: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- **Hopper Architecture**: [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core)
- **FP8 Training**: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)

---

## 总结

Flash Attention 的三个版本体现了深度学习系统优化的演进方向:

1. **v1**: 算法层面创新 (IO-aware)
2. **v2**: 实现层面优化 (并行化、work partitioning)
3. **v3**: 硬件-软件协同设计 (Hopper 特性、FP8)

**选择建议**:
- **A100 及以下**: 使用 Flash Attention v2
- **H100/H200**: 使用 Flash Attention v3
- **长序列训练**: v2/v3 都有显著提升
- **推理优化**: 考虑 v3 的 FP8 支持

**未来方向**:
- 更细粒度的 sparsity 支持
- 多模态 attention 优化
- 自适应精度选择
- 跨节点的分布式 attention

---

**文档维护**:
- 创建时间: 2026-02-02
- 最后更新: 2026-02-02
- 维护者: HPC Learning Repository
