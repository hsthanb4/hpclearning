// ============================================================================
// practice_01_elementwise.cu  Elementwise / 映射类算子（热身）
// ----------------------------------------------------------------------------
// 技能点：一维 grid/block 映射、grid-stride loop、合并访存、launch 参数
//
// 包含练习：
//   01  Vector Add   c[i] = a[i] + b[i]
//   11  SiLU         y = x / (1 + exp(-x))
//
// 学习目标：练熟"idx/stride/loop/launch"四件套，后续所有算子都建立在这上面。
// ============================================================================

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// 练习 01：Vector Add   c[i] = a[i] + b[i]
// ----------------------------------------------------------------------------
// 关键思路：
//   - 一维 grid/block 映射：idx = blockIdx.x * blockDim.x + threadIdx.x
//   - grid-stride loop：kernel 可处理任意 n，且相邻线程读相邻地址（合并访存）
//   - 限制 grid 上限是常见防御性写法
// ============================================================================

__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    // TODO 1: 计算当前线程的全局下标 idx（blockIdx/blockDim/threadIdx）

    // TODO 2: 计算 stride = blockDim.x * gridDim.x

    // TODO 3: 用 grid-stride loop 完成 c[i] = a[i] + b[i]

}

void launch_vector_add(
    const float* a,
    const float* b,
    float* c,
    int n,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;

    // TODO 4: 计算 grid 大小（向上取整）并限制上限 4096

    // TODO 5: 以 4 参形式 launch kernel（grid, block, sharedMem=0, stream）
}

// 问答题 01.1：为什么 grid-stride loop 比"每个线程只处理一个元素"更好？
//       提示：从 kernel 通用性、n 很大时的线程复用、以及拆 grid 上限三方面说。
// 问答题 01.2：为什么 idx 连续的线程读写是 coalesced？如果改成 idx = threadIdx.x 会怎样？
// 问答题 01.3：BLOCK_SIZE=256、n=100000 时 grid 是多少？最后一个 block 有多少线程"空转"？

// ============================================================================
// 练习 11：SiLU / Swish   y = x / (1 + exp(-x))
// ----------------------------------------------------------------------------
// 关键思路：
//   - elementwise kernel：grid-stride loop + 合并访存
//   - 计算瓶颈是 expf；LLM 里 SiLU 常见于 SwiGLU（silu(gate)*up）
// ============================================================================

__global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    // TODO 6: idx / stride / grid-stride loop
    // TODO 7: output[i] = x / (1 + exp(-x))
}

void launch_silu(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    // TODO 8: grid 计算 + 上限 4096 + launch
}

// 问答题 11.1：elementwise kernel 的优化点通常在哪？如果显存带宽是瓶颈，怎么验证？
// 问答题 11.2：expf 是慢指令。如果要进一步加速 SiLU，有哪些思路（近似、查表、融合到 GEMM）？
// 问答题 11.3：SwiGLU 里 silu(gate)*up 为什么要融合成一个 kernel，而不是先算 silu 再乘？
