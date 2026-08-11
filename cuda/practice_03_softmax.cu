// ============================================================================
// practice_03_softmax.cu  Softmax 专题（数值稳定性 + online 合并）
// ----------------------------------------------------------------------------
// 技能点：减 max 防溢出、行级并行、online (m,d) 状态合并、
//         (m,d) 的 shuffle reduce —— FlashAttention 的前置知识
//
// 包含练习：
//   06  Softmax Naive   一线程一输出，每行 O(N^2) 访存（先写低效版）
//   07  Online Softmax  一个 block 一行，2-pass，一次扫描出 max+sum
//
// 学习目标：先理解 naive 为什么慢/不稳，再理解 online 合并公式
// d = d1*exp(m1-m) + d2*exp(m2-m)，这是 FlashAttention rescale 的雏形。
// ============================================================================

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// 练习 06：Softmax Naive  一个线程算一个输出，但每行扫两遍求 max/sum
// ----------------------------------------------------------------------------
// 关键思路：
//   - 先求行 max 防 exp 溢出，再求 sum(exp(x-max))，最后归一化
//   - 低效点：每个输出元素线程都重扫整行 -> 每行 O(N^2) 访存
// ============================================================================

__global__ void softmax_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // TODO 1: 用 2D grid/block 映射出 (row, col)（blockIdx.x/y, blockDim.x/y, threadIdx.x/y）

    if (row < M && col < N) {
        float max_val = -FLT_MAX;

        // TODO 2: 第一次扫整行求 max（防溢出）

        float sum_exp = 0.0f;

        // TODO 3: 第二次扫整行累加 exp(x - max_val)

        // TODO 4: 写 output[row][col] = exp(x-max)/sum_exp
    }
}

void launch_softmax_naive(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    // TODO 5: 计算 2D grid：(N+15)/16 x (M+15)/16
    // TODO 6: launch
}

// 问答题 06.1：为什么必须减 max？不减会发生什么（给一个具体数值例子）？
// 问答题 06.2：这个实现的每行时间复杂度是多少？为什么说它是 O(N^2) 访存？
// 问答题 06.3：2D block(16,16) 有什么问题？为什么通常一行用一个 block 更合理？

// ============================================================================
// 练习 07：Online Softmax（2-pass）  一个 block 负责一行
// ----------------------------------------------------------------------------
// 关键思路：
//   - online 状态 (m, d)：m 是见过的最大值，d = sum(exp(x_i - m))
//   - 合并两个状态：m=max(m1,m2)，d = d1*exp(m1-m) + d2*exp(m2-m)
//   - pass1 一次扫描同时得到行 max 和 denominator；pass2 写输出
// ============================================================================

namespace {

__device__ __forceinline__ void online_combine(
    float& m1,
    float& d1,
    float m2,
    float d2
) {
    // TODO 7: m = fmaxf(m1, m2)
    // TODO 8: d1 = d1*expf(m1-m) + d2*expf(m2-m)；m1 = m
}

__device__ __forceinline__ void warp_reduce_online(float& m, float& d) {
    unsigned mask = 0xffffffff;
    // TODO 9: offset 16->1，shuffle 出 (m2,d2) 后 online_combine
}

template<int BLOCK_SIZE>
__global__ void softmax_online_2pass(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    // Pass 1：stride 扫自己的列，把 (x, 1) 与当前状态合并
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];
        // TODO 10: 用 online_combine 合并 (local_m, local_d) 与 (x, 1)
    }

    // TODO 11: warp_reduce_online(local_m, local_d)

    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_d[NUM_WARPS];

    int lane = tid & 31;
    int warp = tid >> 5;

    // TODO 12: lane==0 写入 smem_m/smem_d[warp]，__syncthreads()

    float row_m = -FLT_MAX;
    float row_d = 0.0f;
    if (warp == 0) {
        // TODO 13: lane<NUM_WARPS 时读回 smem，warp_reduce_online，lane==0 写回 smem[0]
    }
    __syncthreads();

    // TODO 14: 读 row_m/row_d，Pass 2 写 exp(x-row_m)/row_d
}

} // namespace

void launch_softmax_online_2pass(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 15: launch，grid=M
}

// 问答题 07.1：online_combine 的推导依据是什么？为什么 d 要按新旧 m 缩放？
// 问答题 07.2：为什么这个版本比 naive 快 O(N) 倍？它的访存次数是多少？
// 问答题 07.3：online softmax 是 FlashAttention 的核心。如果分片 (m1,d1),(m2,d2) 来自
//       不同的数据分块，合并时哪一步对应 FlashAttention 的 rescale？
