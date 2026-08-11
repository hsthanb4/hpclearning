// ============================================================================
// practice_04_tiling.cu  Shared Memory Tiling 专题（转置 + GEMM）
// ----------------------------------------------------------------------------
// 技能点：naive 的访存缺陷 -> shared memory tile 缓存复用、
//         __syncthreads 的正确位置、bank conflict 与 padding、
//         2D tile 坐标映射
//
// 包含练习：
//   08  Transpose Naive   一线程搬一个元素（先写低效版）
//   09  Transpose Tiled   32x32 tile + 33 列 padding
//   10  GEMM Tiled        C = A*B，16x16 tile，每线程一个输出
//
// 学习目标：理解"用 shared memory 把 strided 访存变成合并访存"和
// "数据复用减少 global 访存"两条主线，以及同步的代价。
// ============================================================================

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

// ============================================================================
// 练习 08：Transpose Naive  一线程搬一个元素
// ----------------------------------------------------------------------------
// 关键思路：
//   - 读 input 合并；写 output[col*M+row] 跨距 M，不合并
//   - 瓶颈是 strided global store
// ============================================================================

__global__ void transpose_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // TODO 1: 2D 映射出 (row, col)，越界 return
    // TODO 2: output[col * M + row] = input[row * N + col]
}

void launch_transpose_naive(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    // TODO 3: 2D grid 并 launch
}

// 问答题 08.1：读是合并的、写是跨步的，为什么说写是主要瓶颈？
//       提示：比较 32B/128B 缓存行的利用率。
// 问答题 08.2：如果 M=N=32，一个 warp 写 output 时覆盖多少个不同的 cache line？
// 问答题 08.3：transpose 有没有可能做到读写都合并？需要借助什么（提示：shared memory）？

// ============================================================================
// 练习 09：Transpose Tiled  32x32 shared memory tile + 33 列 padding
// ----------------------------------------------------------------------------
// 关键思路：
//   - 合并读入 shared memory tile，转置后再合并写回
//   - tile 第二维用 33（TILE_DIM+1），消除 tile[col][row] 的 32 路 bank conflict
//   - 交换 blockIdx.x/y 实现 tile 级转置
// ============================================================================

namespace {

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // TODO 4: 声明 __shared__ float tile[TILE_DIM][TILE_DIM + 1];  （为什么 +1？）

    // TODO 5: x = blockIdx.x*TILE_DIM + threadIdx.x; y = blockIdx.y*TILE_DIM + threadIdx.y

    // TODO 6: j 循环（j += BLOCK_ROWS）把 input 的 32x32 块读入 tile，越界跳过

    // TODO 7: __syncthreads() 保证 tile 写完再读

    // TODO 8: 交换 blockIdx.x/y 计算写回坐标（x,y 重新赋值）

    // TODO 9: j 循环从 tile[threadIdx.x][threadIdx.y+j] 写回 output，越界跳过
}

} // namespace

void launch_transpose_tiled(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    // TODO 10: block(TILE_DIM, BLOCK_ROWS)；grid((N+31)/32, (M+31)/32)；launch
}

// 问答题 09.1：shared memory 为什么会有 bank conflict？tile[32][32] 与 tile[32][33]
//       在访问 tile[col][row] 时冲突次数有何不同（画一下 row/col 的 bank 分布）？
// 问答题 09.2：为什么写回时交换 blockIdx.x 和 blockIdx.y？如果不交换会怎样？
// 问答题 09.3：TILE_DIM=32 时一个 block 只有 32x8=256 线程，为什么用 j 循环搬 4 行
//       而不是直接开 1024 线程？

// ============================================================================
// 练习 10：GEMM Tiled   C = A*B，A:[M,K] B:[K,N] C:[M,N]，每线程算一个输出
// ----------------------------------------------------------------------------
// 关键思路：
//   - naive GEMM 每个 C 元素重读一整行 A + 一整列 B
//   - shared memory tile 缓存 A/B 子块，tile 内线程复用
//   - K 维分块推进，每轮 load + 双 __syncthreads
// ============================================================================

namespace {

constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    // TODO 11: 声明 __shared__ float As[TILE][TILE], Bs[TILE][TILE]

    // TODO 12: row = blockIdx.y*TILE + threadIdx.y; col = blockIdx.x*TILE + threadIdx.x

    float acc = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        // TODO 13: As[threadIdx.y][threadIdx.x] = (row<M && k0+tx<K) ? A[row*K+k0+tx] : 0
        // TODO 14: Bs[threadIdx.y][threadIdx.x] = (k0+ty<K && col<N) ? B[(k0+ty)*N+col] : 0

        // TODO 15: __syncthreads()

        // TODO 16: 内层 k 循环 acc += As[ty][k] * Bs[k][tx]

        // TODO 17: __syncthreads()（为什么这第二个同步必不可少？）
    }

    // TODO 18: row<M && col<N 时写 C[row*N+col] = acc
}

} // namespace

void launch_gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    // TODO 19: block(TILE,TILE)；grid((N+15)/16, (M+15)/16)；launch
}

// 问答题 10.1：为什么需要两个 __syncthreads()？去掉第二个会有什么后果（数据竞争在哪）？
// 问答题 10.2：shared memory tiling 减少了几倍 global 访存？（对比每线程直接读）
// 问答题 10.3：这个版本每个线程只算一个 C 元素。要提升性能你会怎么改？
//       提示：寄存器 tile、每线程多元素、向量化加载、cp.async、double buffering。
