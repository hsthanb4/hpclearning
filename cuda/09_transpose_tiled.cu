#include <cuda_runtime.h>

namespace {

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

// Tiled Transpose:
// input:  [M, N] row-major
// output: [N, M] row-major
//
// naive transpose 的问题：
// 读 input 是连续的，但写 output 会变成大步长 strided write，
// 一个 warp 写出去的地址不连续，global memory store 不合并。
//
// 优化思路：
// 1. 先把 input 的 32x32 tile 合并读取到 shared memory。
// 2. 再从 shared memory 转置后合并写到 output。
// 3. shared memory 第二维用 33，避免 tile[col][row] 访问时 32 路 bank conflict。
__global__ void transpose_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 一个 block 只有 32x8=256 个线程。
    // 每个线程用 j 循环搬 4 行，合起来覆盖 32x32 tile。
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < M && col < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * N + col];
        }
    }

    __syncthreads();

    // 交换 blockIdx.x 和 blockIdx.y，实现 tile 级转置后的写回位置。
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < N && col < M) {
            output[row * M + col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

} // namespace

void launch_transpose_tiled(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    transpose_tiled_kernel<<<grid, block, 0, stream>>>(input, output, M, N);
}
