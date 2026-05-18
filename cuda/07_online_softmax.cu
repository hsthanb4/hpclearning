#include <float.h>
#include <cuda_runtime.h>

// Online Softmax 里每个局部状态用两个数表示：
// m = 当前看过元素的最大值
// d = sum(exp(x_i - m))
//
// 如果两个分片状态分别是 (m1, d1) 和 (m2, d2)，合并时新的最大值是：
// m = max(m1, m2)
//
// 原来的 d1 基于 m1，d2 基于 m2；合并到新 m 后需要重新缩放：
// d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
//
// 这个合并公式是 online softmax 的核心。
__device__ __forceinline__ void online_combine(
    float& m1,
    float& d1,
    float m2,
    float d2
) {
    float m = fmaxf(m1, m2);
    d1 = d1 * expf(m1 - m) + d2 * expf(m2 - m);
    m1 = m;
}

// warp 内把 32 个线程各自的 (m, d) 合并成一个状态。
// shuffle_down 每次把 offset 位置的线程状态读过来，然后用 online_combine 合并。
__device__ __forceinline__ void warp_reduce_online(float& m, float& d) {
    unsigned mask = 0xffffffff;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(mask, m, offset);
        float d2 = __shfl_down_sync(mask, d, offset);
        online_combine(m, d, m2, d2);
    }
}

// Softmax: 对 [M, N] 的每一行做 softmax。
//
// 传统稳定 softmax 通常是 3 pass：
// 1. 求 max
// 2. 求 sum(exp(x - max))
// 3. 写 output
//
// 这个版本是 2 pass online softmax：
// 1. 一次扫描同时得到整行 max 和 denominator
// 2. 第二次扫描写 output
//
// 一个 block 负责一行。这样一行的 max/sum 只算一次，
// 避免“每个输出元素都重复扫整行”的 O(N^2) 写法。
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

    // 每个线程维护自己负责列的 online 状态。
    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    // Pass 1：每个线程 stride 扫一部分列。
    // 对新元素 x，把当前状态 (local_m, local_d) 和单元素状态 (x, 1) 合并。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];

        float new_m = fmaxf(local_m, x);
        local_d = local_d * expf(local_m - new_m) + expf(x - new_m);
        local_m = new_m;
    }

    // 先在每个 warp 内合并 local 状态。
    warp_reduce_online(local_m, local_d);

    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;

    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_d[NUM_WARPS];

    int lane = tid & 31;
    int warp = tid >> 5;

    // 每个 warp 的 lane 0 把该 warp 的结果放到 shared memory。
    if (lane == 0) {
        smem_m[warp] = local_m;
        smem_d[warp] = local_d;
    }

    __syncthreads();

    // warp 0 负责把所有 warp 的 partial 状态合成整行状态。
    float row_m = -FLT_MAX;
    float row_d = 0.0f;

    if (warp == 0) {
        if (lane < NUM_WARPS) {
            row_m = smem_m[lane];
            row_d = smem_d[lane];
        }

        warp_reduce_online(row_m, row_d);

        // 把最终整行 max 和 denominator 广播用的数据写回 smem[0]。
        if (lane == 0) {
            smem_m[0] = row_m;
            smem_d[0] = row_d;
        }
    }

    __syncthreads();

    row_m = smem_m[0];
    row_d = smem_d[0];

    // Pass 2：整行 max 和 denominator 已知后，写最终 softmax。
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        row_out[col] = expf(row_in[col] - row_m) / row_d;
    }
}

void launch_softmax_online_2pass(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    softmax_online_2pass<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        input, output, M, N
    );
}
