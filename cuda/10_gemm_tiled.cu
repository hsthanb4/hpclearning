#include <cuda_runtime.h>

namespace {

constexpr int TILE = 16;

// Tiled GEMM: C = A * B
// A: [M, K] row-major
// B: [K, N] row-major
// C: [M, N] row-major
//
// 面试里的重点不是打赢 cuBLAS，而是讲清楚：
// 1. naive GEMM 每个 C 元素都要读一整行 A 和一整列 B。
// 2. shared memory tiling 把 A/B 子块缓存下来，让一个 tile 内多个线程复用。
// 3. 每个 thread 负责一个 C[row, col]，在 K 维上分块累加。
//
// 这是精简版：每线程算一个输出元素，TILE=16。
// 更高性能版本会让每线程算多个元素、用寄存器 tile、向量化、cp.async 等。
__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // 沿 K 维一块一块推进。
    // 每一轮加载 A 的 [row, k_tile:k_tile+TILE)
    // 和 B 的 [k_tile:k_tile+TILE, col] 到 shared memory。
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int a_col = k0 + threadIdx.x;
        int b_row = k0 + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // 下一轮会覆盖 As/Bs，所以所有线程必须先算完当前 tile。
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
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
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
