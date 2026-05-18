#include <cuda_runtime.h>

namespace {

// 面试重点 1：warp 内 reduce 不需要 shared memory。
// 一个 warp 内 32 个线程可以通过 shuffle 指令直接交换寄存器里的值，
// 比写 shared memory 再同步更轻。
template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 面试重点 2：block reduce 通常分两层：
// 1. 每个 warp 先在寄存器里 reduce 出一个 partial sum。
// 2. 每个 warp 的 lane 0 把 partial sum 写入 shared memory。
// 3. warp 0 再把这些 warp partial sum reduce 成 block 级结果。
template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    val = warp_reduce_sum<BLOCK_SIZE>(val);

    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    // 只有前 NUM_WARPS 个线程需要读取 shared memory。
    // 这些线程都在 warp 0 里，因此后续仍然可以用 warp reduce。
    val = (threadIdx.x < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }

    // 这个同步保证同一个 kernel 里连续调用 block_reduce_sum 时，
    // 下一次调用不会覆盖上一次还没读完的 smem。
    __syncthreads();
    return val;
}

// LayerNorm: 对每一行独立做归一化。
// input/output: [M, N] row-major
// gamma/beta: [N]
//
// 数学公式：
// mean = sum(x_i) / N
// var  = sum(x_i^2) / N - mean^2
// y_i  = (x_i - mean) / sqrt(var + eps) * gamma_i + beta_i
//
// 优化思路：
// 一个 block 负责一整行，block 内线程并行扫 N 个元素。
// 每个线程先做局部累加，再通过 warp reduce + smem 得到整行统计量。
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

    // 每个线程用 stride 方式处理多个列。
    // col 连续分配给 threadIdx.x，能保证同一轮访问时线程读连续地址，
    // 也就是 coalesced global memory load。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];
        local_sum += x;
        local_sq_sum += x * x;
    }

    // 两个 reduce 分别得到 sum(x) 和 sum(x^2)。
    float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
    float sq_sum = block_reduce_sum<BLOCK_SIZE>(local_sq_sum);

    // mean 和 inv_std 是整行共享的标量。
    // 只让 tid 0 计算一次，然后放到 shared memory 广播给全 block。
    __shared__ float smem_mean;
    __shared__ float smem_inv_std;

    if (tid == 0) {
        float mean = sum / static_cast<float>(N);
        float var = sq_sum / static_cast<float>(N) - mean * mean;

        // fmaxf 防止浮点舍入导致 var 变成很小的负数。
        smem_mean = mean;
        smem_inv_std = rsqrtf(fmaxf(var, 0.0f) + eps);
    }
    __syncthreads();

    float mean = smem_mean;
    float inv_std = smem_inv_std;

    // 第二次扫这一行，写最终归一化结果。
    // LayerNorm 至少需要先知道整行 mean/var，所以写输出通常是第二 pass。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float y = (row_in[col] - mean) * inv_std;
        row_out[col] = y * gamma[col] + beta[col];
    }
}

} // namespace

// 简洁 launch wrapper：面试时可以先固定 BLOCK_SIZE=256，
// 后续再讨论如何根据 N 和 occupancy 调参。
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
    layernorm_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        input, gamma, beta, output, M, N, eps
    );
}
