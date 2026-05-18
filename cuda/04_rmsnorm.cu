#include <cuda_runtime.h>

namespace {

// warp reduce: 用 shuffle 在寄存器之间传值，避免 shared memory 往返。
template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// block reduce: warp 内 reduce + shared memory 跨 warp 汇总。
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

    val = (threadIdx.x < NUM_WARPS) ? smem[lane] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }

    __syncthreads();
    return val;
}

// RMSNorm: 和 LayerNorm 很像，但不减 mean，也没有 beta。
// input/output: [M, N] row-major
// weight: [N]
//
// 数学公式：
// rms = sqrt(sum(x_i^2) / N + eps)
// y_i = x_i / rms * weight_i
//
// 面试里常问它为什么比 LayerNorm 少一些计算：
// RMSNorm 不需要 sum(x)，只需要 sum(x^2)，所以少一次 reduce，
// 同时输出阶段也少了减 mean 和 beta。
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

    // 每个线程处理若干列，局部累加平方和。
    // 同一轮 col=tid 时，相邻线程访问相邻地址，访存合并。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];
        local_sq_sum += x * x;
    }

    float sq_sum = block_reduce_sum<BLOCK_SIZE>(local_sq_sum);

    // inv_rms 是这一行所有元素共享的标量。
    __shared__ float smem_inv_rms;

    if (tid == 0) {
        smem_inv_rms = rsqrtf(sq_sum / static_cast<float>(N) + eps);
    }
    __syncthreads();

    float inv_rms = smem_inv_rms;

    // 第二 pass 写输出。这里读 input、读 weight、写 output 都是连续访问。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        row_out[col] = row_in[col] * inv_rms * weight[col];
    }
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
    rmsnorm_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        input, weight, output, M, N, eps
    );
}
