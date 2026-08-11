// ============================================================================
// practice_answer_04_tiling.cu
// 对应练习文件：practice_04_tiling.cu（先做填空，再对照本答案）
// ============================================================================

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// 练习 08：Transpose Naive —— 完整答案
// ----------------------------------------------------------------------------
__global__ void transpose_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // TODO 1/2: 2D 映射 + 转置写回
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

void launch_transpose_naive(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0) return;

    dim3 block(16, 16);
    // TODO 3: 2D grid + launch
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, M, N);
}

// 问答题 08.1 答案：读取地址连续，只需少量内存事务；写地址相隔 M 个
//   float，会触发更多 32B sector/128B cache line，有效带宽更低。
// 问答题 08.2 答案：本代码的 block 是 16×16，一个 warp 包含两行、每行 16 个线程。
//   M=N=32 且地址对齐时，它写入 16 个不同的 128B cache line（也是 16 个 32B sector）。
// 问答题 08.3 答案：可以。先合并读入 shared-memory tile，再从转置坐标取数并
//   合并写回；padding 用于避免 shared-memory bank conflict。

// ----------------------------------------------------------------------------
// 练习 09：Transpose Tiled —— 完整答案
// ----------------------------------------------------------------------------
namespace p09 {

constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ void transpose_tiled_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // TODO 4: 33 列 padding 防 bank conflict（见问答题 09.1）
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // TODO 5: 当前 block 负责的 input tile 坐标
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // TODO 6: j 循环搬 4 行（BLOCK_ROWS=8，256 线程覆盖 32x32）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < M && col < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[row * N + col];
        }
    }

    // TODO 7: 写完才能读（跨线程共享数据必须同步）
    __syncthreads();

    // TODO 8: 交换 blockIdx.x/y：tile 转置后写到 output 的对应位置
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // TODO 9: 从 smem 转置读、合并写回 output（边界换成 N/M）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        int row = y + j;
        int col = x;
        if (row < N && col < M) {
            output[row * M + col] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

} // namespace p09

void launch_transpose_tiled(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0) return;

    // TODO 10: block(32, 8)；grid 按 TILE_DIM 向上取整
    dim3 block(p09::TILE_DIM, p09::BLOCK_ROWS);
    dim3 grid((N + p09::TILE_DIM - 1) / p09::TILE_DIM,
              (M + p09::TILE_DIM - 1) / p09::TILE_DIM);
    p09::transpose_tiled_kernel<<<grid, block, 0, stream>>>(input, output, M, N);
}

// 问答题 09.1 答案：对 4B float，bank 约为 word_address%32。按列读
//   tile[32][32] 时 32 个 lane 落在同一 bank，形成 32 路冲突；stride 改为 33 后
//   bank=(lane*33+c)%32，32 个 lane 分布到 32 个 bank。
// 问答题 09.2 答案：input tile (bx,by) 转置后应写到 output tile (by,bx)。
//   不交换只会在原 tile 位置做局部转置，非对角 tile 不会互换。
// 问答题 09.3 答案：1024 线程是 block 上限，会限制驻留 block 数和调度灵活性。
//   256 线程循环搬 4 行仍能合并访存，通常更易获得良好并发度。

// ----------------------------------------------------------------------------
// 练习 10：GEMM Tiled —— 完整答案
// ----------------------------------------------------------------------------
namespace p10 {

constexpr int TILE = 16;

__global__ void gemm_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    // TODO 11: A/B 各一个 smem tile
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // TODO 12: 当前线程负责的 C 元素
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // 沿 K 维分块推进
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // TODO 13/14: 加载 A/B 子块到 smem，越界填 0（等价于跳过该乘积）
        int a_col = k0 + threadIdx.x;
        int b_row = k0 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // TODO 15: 同步：所有线程的 load 完成才能开始读
        __syncthreads();

        // TODO 16: 内层 K 维点积（从 smem 读，快）
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // TODO 17: 同步：所有线程算完当前 tile 才能覆盖写 As/Bs
        __syncthreads();
    }

    // TODO 18: 写回（越界不写）
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

} // namespace p10

void launch_gemm_tiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K < 0) return;

    // TODO 19: block(16,16)；grid 按输出 tile 数
    dim3 block(p10::TILE, p10::TILE);
    dim3 grid((N + p10::TILE - 1) / p10::TILE, (M + p10::TILE - 1) / p10::TILE);
    p10::gemm_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

// 问答题 10.1 答案：第一个同步保证 tile 已全部写入再读；第二个保证当前
//   tile 已全部读完再覆盖。去掉第二个会产生跨迭代的读写竞争。
// 问答题 10.2 答案：忽略边界和 cache，naive 读 2MNK 个 float。TILE=T 时，
//   tiled 约读 2MNK/T 个 float，因为每次读入的 A/B 元素在 tile 内复用 T 次。
//   因此本例约减少 16 倍，而不是全矩阵每个元素只读一次。
// 问答题 10.3 答案：让每线程计算多个输出形成寄存器 tile；再配合向量化加载、
//   双缓冲和 cp.async 隐藏访存延迟。低精度场景可进一步使用 tensor core。
