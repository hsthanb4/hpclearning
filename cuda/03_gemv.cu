#include <cuda_runtime.h>

namespace {

// GEMV 的核心是 dot product，因此 reduce 是主要优化点。
template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 把一个 block 内所有线程的局部 dot partial sum 合并成一个 sum。
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];

    int lane = threadIdx.x & 31;  //等价于 threadIdx.x % 32
    int warp = threadIdx.x >> 5;  //等价于 threadIdx.x / 32

    val = warp_reduce_sum<BLOCK_SIZE>(val);

    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    val = (threadIdx.x < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }

    __syncthreads();
    return val;
}

// GEMV: y = A * x + bias
// A: [M, N] row-major
// x: [N]
// y/bias: [M]
//
// 简化优化版本：
// 一个 block 负责 A 的一行，block 内线程并行做 dot product。
// 这适合面试讲清楚线程映射、coalesced load、block reduce。
//
// 局限：
// x 会被每一行重复读取。更进一步的优化可以把 x 分块缓存到 shared memory，
// 或者多个 row/block 复用 x tile，但代码会复杂很多。
template<int BLOCK_SIZE>
__global__ void gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int M,
    int N
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_A = A + row * N;
    float local_sum = 0.0f;

    // 每个线程负责这一行里的若干列。
    // A[row, col] 是连续地址，因此一个 warp 同一轮读 A 时是合并访存。
    // x[col] 也是连续读取，但不同 row 会重复读同一个 x。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        local_sum += row_A[col] * x[col];
    }

    float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    // reduce 完后，只有一个线程写 y[row]，避免写冲突。
    if (tid == 0) {
        y[row] = sum + (bias ? bias[row] : 0.0f);
    }
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
    gemv_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        A, x, bias, y, M, N
    );
}
