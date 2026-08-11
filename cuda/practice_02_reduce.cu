// ============================================================================
// practice_02_reduce.cu  Reduce 类算子（warp/block reduce 核心技能）
// ----------------------------------------------------------------------------
// 技能点：shuffle reduce、shared memory 跨 warp 归并、两阶段 kernel、
//         (value, index) 对同步 reduce
//
// 包含练习：
//   02  Reduce Sum   两阶段求和
//   03  GEMV          y = A*x + bias（行并行 + dot reduce）
//   04  RMSNorm       只统计 sum(x^2)
//   05  LayerNorm     统计 sum(x) 与 sum(x^2)
//   12  ArgMax        两阶段 (val, idx) reduce
//
// 学习目标：把 warp_reduce_sum / block_reduce_sum 练到能默写，
// 然后理解它们如何被复用/变形（RMSNorm 少一个统计、ArgMax 多传一个 index）。
// ============================================================================

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// 练习 02：Reduce Sum  对 input[0:n] 求和（两阶段归约）
// ----------------------------------------------------------------------------
// 关键思路：
//   - 每个线程先 grid-stride 累加多个元素（寄存器累加便宜）
//   - warp 内 shuffle reduce（不需要 shared memory）
//   - 跨 warp 用 shared memory 存每个 warp 的 partial，再归并
// ============================================================================

namespace {

template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // TODO 1: 用 __shfl_down_sync(0xffffffff, val, offset) 从 offset=16 循环到 1

    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    // TODO 2: 先做 warp_reduce_sum

    // TODO 3: lane==0 时把结果写入 smem[warp]

    // TODO 4: __syncthreads() 后，warp 0 读回所有 partial 再 reduce 一次

    return val;
}

template<int BLOCK_SIZE>
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial,
    int n
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // TODO 5: grid-stride loop 累加 local_sum

    float block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    // TODO 6: tid==0 时把 block_sum 写入 partial[blockIdx.x]
}

} // namespace

void launch_reduce_sum_stage(
    const float* input,
    float* partial,
    int n,
    int num_blocks,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 7: launch reduce_sum_kernel，grid=num_blocks
}

// 问答题 02.1：__shfl_down_sync 的第一个参数 0xffffffff 是什么？为什么需要它？
// 问答题 02.2：为什么两阶段（每 block 出 partial，再归并 partial）比单 kernel 全部归并更好？
//       提示：从 block 间无法同步、原子操作竞争、以及第二次 launch 的成本讨论。
// 问答题 02.3：block_reduce_sum 里为什么先 warp reduce 再写 shared memory，而不是直接写？

// ============================================================================
// 练习 03：GEMV   y = A * x + bias，A:[M,N] row-major
// ----------------------------------------------------------------------------
// 关键思路：
//   - 一个 block 负责 A 的一行，block 内线程并行做 dot product
//   - A[row, col] 连续地址 -> 合并访存；x 会被每行重复读（局限点）
// ============================================================================

namespace {

// TODO 8: 复用练习 02 的 warp_reduce_sum 与 block_reduce_sum（直接照抄）

template<int BLOCK_SIZE>
__global__ void gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int M,
    int N
) {
    // TODO 9: row = blockIdx.x，tid = threadIdx.x，越界 row>=M 直接 return

    const float* row_A = A + row * N;
    float local_sum = 0.0f;

    // TODO 10: 每个线程 stride 循环累加 row_A[col] * x[col]

    float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    // TODO 11: tid==0 写 y[row] = sum + bias（bias 可为空）
}

} // namespace

void launch_gemv(
    const float* A,
    const float* x,
    const float* bias,
    float* y,
    int M,
    int N,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 12: launch，grid=M（一个 block 一行）
}

// 问答题 03.1：为什么 x 会被重复读 M 次？可以怎么优化（提示：shared memory tile / 多行复用）？
// 问答题 03.2：这个 kernel 的访存模式里，A 是合并的，为什么 x 的读取不是瓶颈？
//       提示：考虑 cache 命中率与 M、N 的相对大小。
// 问答题 03.3：如果 N < BLOCK_SIZE，会发生什么？这种形状下什么分配策略更好？

// ============================================================================
// 练习 04：RMSNorm   y_i = x_i / sqrt(sum(x^2)/N + eps) * weight_i
// ----------------------------------------------------------------------------
// 关键思路：
//   - 只需 sum(x^2) 一次 reduce，不需要减 mean（比 LayerNorm 少一次统计）
//   - inv_rms 是整行共享标量：tid 0 计算后放 shared memory 广播
//   - 第二 pass 写输出
// ============================================================================

namespace {

// TODO 13: 复用 warp_reduce_sum 与 block_reduce_sum

template<int BLOCK_SIZE>
__global__ void rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int M,
    int N,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float local_sq_sum = 0.0f;

    // TODO 14: stride 循环累加 x*x 到 local_sq_sum

    float sq_sum = block_reduce_sum<BLOCK_SIZE>(local_sq_sum);

    // TODO 15: tid==0 时计算 smem_inv_rms = rsqrtf(sq_sum/N + eps)，然后 __syncthreads()

    // TODO 16: 第二 pass 写 row_out[col] = row_in[col] * inv_rms * weight[col]
}

} // namespace

void launch_rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    int M,
    int N,
    float eps,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 17: launch，grid=M
}

// 问答题 04.1：RMSNorm 比 LayerNorm 少了哪些计算？为什么 LLM 常用 RMSNorm 而不是 LayerNorm？
// 问答题 04.2：为什么 inv_rms 要先广播到 shared memory 而不是每个线程各自算一遍？
// 问答题 04.3：这个实现读 input 两次（pass1 统计、pass2 归一化）。如果 N 大到超出一行 cache，
//       能不能用一次扫描完成？代价是什么（提示：需要保存整行 x 到哪？）？

// ============================================================================
// 练习 05：LayerNorm   y_i = (x_i - mean)/sqrt(var+eps) * gamma_i + beta_i
// ----------------------------------------------------------------------------
// 关键思路：
//   - 两个 reduce 分别得到 sum(x) 与 sum(x^2)，再合成 mean/var
//   - fmaxf(var, 0) 防浮点舍入导致负方差
//   - 第二 pass 归一化
// ============================================================================

namespace {

// TODO 18: 复用 warp_reduce_sum 与 block_reduce_sum

template<int BLOCK_SIZE>
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int M,
    int N,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // TODO 19: stride 循环同时累加 local_sum 与 local_sq_sum

    float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
    float sq_sum = block_reduce_sum<BLOCK_SIZE>(local_sq_sum);

    __shared__ float smem_mean;
    __shared__ float smem_inv_std;

    if (tid == 0) {
        // TODO 20: mean = sum/N；var = sq_sum/N - mean^2（用 fmaxf 防负）
        //          smem_inv_std = rsqrtf(fmaxf(var,0)+eps)
    }
    __syncthreads();

    // TODO 21: 第二 pass 写 (x-mean)*inv_std*gamma + beta
}

} // namespace

void launch_layernorm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int M,
    int N,
    float eps,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 22: launch，grid=M
}

// 问答题 05.1：var 用 E[x^2] - E[x]^2 计算在数值上有什么风险？更稳的算法是什么？
// 问答题 05.2：kernel 里连续调用了两次 block_reduce_sum。为什么第二次调用前不需要担心
//       shared memory 残留（提示：函数末尾的 __syncthreads 在防什么）？
// 问答题 05.3：mean/var 只需一个线程算，为什么要用 shared memory 广播而不是直接传寄存器？

// ============================================================================
// 练习 12：ArgMax  两阶段：每 block 出 (val, idx)，再归并
// ----------------------------------------------------------------------------
// 关键思路：
//   - reduce 时同时传递 (value, index) 对，shuffle 传 value 比较、传 index 选择
//   - 第一轮每个 block 出一个局部最大；第二轮一个 block 归并
// ============================================================================

namespace {

struct ValIdx {
    float val;
    int idx;
};

template<int BLOCK_SIZE>
__device__ __forceinline__ ValIdx warp_reduce_max(ValIdx vi) {
    // TODO 23: offset 16->1，shuffle 出 other_val/other_idx
    // TODO 24: other_val > vi.val 时替换 vi.val 与 vi.idx
    return vi;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ ValIdx block_reduce_max(ValIdx vi) {
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem_val[NUM_WARPS];
    __shared__ int smem_idx[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    // TODO 25: warp_reduce_max，lane==0 写入 smem_val/smem_idx，__syncthreads()

    // TODO 26: warp 0 读回（不足 NUM_WARPS 用 -FLT_MAX/-1 填充）再 reduce
    return vi;
}

template<int BLOCK_SIZE>
__global__ void argmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_val,
    int* __restrict__ partial_idx,
    int n
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    ValIdx local = {-FLT_MAX, -1};

    // TODO 27: grid-stride loop 找自己负责区间的最大值（同步更新 val 与 idx）

    ValIdx result = block_reduce_max<BLOCK_SIZE>(local);

    // TODO 28: tid==0 写 partial_val/partial_idx[blockIdx.x]
}

template<int BLOCK_SIZE>
__global__ void argmax_final_kernel(
    const float* __restrict__ partial_val,
    const int* __restrict__ partial_idx,
    int* __restrict__ output,
    int num_blocks
) {
    int tid = threadIdx.x;

    ValIdx local = {-FLT_MAX, -1};

    // TODO 29: 一个 block 扫所有 partial（stride = BLOCK_SIZE），归并后 *output = result.idx
}

} // namespace

void launch_argmax(
    const float* d_input,
    int* d_output,
    float* d_partial_val,
    int* d_partial_idx,
    int n,
    int num_blocks,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 30: launch argmax_kernel（grid=num_blocks）后 launch argmax_final_kernel（grid=1）
}

// 问答题 12.1：为什么不能只用一个 kernel 一个 block 扫完 1M 元素？（提示：并行度/延迟）
// 问答题 12.2：warp_reduce_max 里 shuffle 同时传 val 和 idx，为什么 idx 的 shuffle 不是浪费？
//       提示：比较"传 index"与"先传 val 再广播 winner lane 的 idx"两种方案。
// 问答题 12.3：如果数组里有两个相等的最大值，这个实现返回哪个索引？这算 bug 吗？怎么稳定化？
