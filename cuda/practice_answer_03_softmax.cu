// ============================================================================
// practice_answer_03_softmax.cu
// 对应练习文件：practice_03_softmax.cu（先做填空，再对照本答案）
// ============================================================================

#include <cuda_runtime.h>
#include <float.h>

// ----------------------------------------------------------------------------
// 练习 06：Softmax Naive —— 完整答案
// ----------------------------------------------------------------------------
__global__ void softmax_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // TODO 1: 2D 映射
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float max_val = -FLT_MAX;

        // TODO 2: 第一次扫整行求 max
        for (int i = 0; i < N; ++i) {
            max_val = fmaxf(max_val, input[row * N + i]);
        }

        float sum_exp = 0.0f;

        // TODO 3: 第二次扫整行求 denominator
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(input[row * N + i] - max_val);
        }

        // TODO 4: 归一化写自己的元素
        output[row * N + col] = expf(input[row * N + col] - max_val) / sum_exp;
    }
}

void launch_softmax_naive(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0) return;

    dim3 block(16, 16);
    // TODO 5/6: 2D grid + launch
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    softmax_naive_kernel<<<grid, block, 0, stream>>>(input, output, M, N);
}

// 问答题 06.1 答案：不减 max 时，expf 在输入约大于 88.7 时溢出。
//   例如 [1000,999] 会得到 inf/inf。减 max 利用 softmax 的平移不变性，使指数自变量不大于 0。
// 问答题 06.2 答案：每个输出都扫整行两遍，每行约读取 2N²+N 次，即
//   O(N²)；整个矩阵为 O(MN²)。
// 问答题 06.3 答案：16×16 block 混合了多行，但代码没有共享行统计，每个线程
//   仍重复求 max 和 sum。通常让一个 block 协作处理一行，统计量只归约一次。

// ----------------------------------------------------------------------------
// 练习 07：Online Softmax（2-pass）—— 完整答案
// ----------------------------------------------------------------------------
namespace p07 {

__device__ __forceinline__ void online_combine(
    float& m1,
    float& d1,
    float m2,
    float d2
) {
    // TODO 7/8: 新 max + 按旧 max 缩放合并 d
    float m = fmaxf(m1, m2);
    d1 = d1 * expf(m1 - m) + d2 * expf(m2 - m);
    m1 = m;
}

__device__ __forceinline__ void warp_reduce_online(float& m, float& d) {
    unsigned mask = 0xffffffff;
    // TODO 9: shuffle 出 (m2, d2) 后 online_combine
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m2 = __shfl_down_sync(mask, m, offset);
        float d2 = __shfl_down_sync(mask, d, offset);
        online_combine(m, d, m2, d2);
    }
}

template<int BLOCK_SIZE>
__global__ void softmax_online_2pass(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024 && BLOCK_SIZE % 32 == 0,
                  "BLOCK_SIZE must contain full warps");
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    // Pass 1：stride 扫自己的列，逐元素合并 online 状态
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];
        // TODO 10: 单元素状态就是 (x, 1)
        online_combine(local_m, local_d, x, 1.0f);
    }

    // TODO 11: warp 内先合并 (m,d)
    warp_reduce_online(local_m, local_d);

    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem_m[NUM_WARPS];
    __shared__ float smem_d[NUM_WARPS];

    int lane = tid & 31;
    int warp = tid >> 5;

    // TODO 12: 每 warp lane0 写 smem
    if (lane == 0) {
        smem_m[warp] = local_m;
        smem_d[warp] = local_d;
    }
    __syncthreads();

    // TODO 13: warp0 归并所有 warp 的 (m,d)，结果写回 smem[0] 广播
    float row_m = -FLT_MAX;
    float row_d = 0.0f;
    if (warp == 0) {
        if (lane < NUM_WARPS) {
            row_m = smem_m[lane];
            row_d = smem_d[lane];
        }
        warp_reduce_online(row_m, row_d);
        if (lane == 0) {
            smem_m[0] = row_m;
            smem_d[0] = row_d;
        }
    }
    __syncthreads();

    row_m = smem_m[0];
    row_d = smem_d[0];

    // Pass 2：整行 max/denominator 已知，写最终 softmax
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        row_out[col] = expf(row_in[col] - row_m) / row_d;
    }
}

} // namespace p07

void launch_softmax_online_2pass(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    if (M <= 0 || N <= 0) return;

    p07::softmax_online_2pass<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        input, output, M, N
    );
}

// 问答题 07.1 答案：合并后用 m=max(m1,m2) 作统一基准。因为
//   exp(x−m)=exp(x−mi)·exp(mi−m)，所以 d=d1·exp(m1−m)+d2·exp(m2−m)。
// 问答题 07.2 答案：naive 每行读 O(N²) 次；online 两遍共读 2N 次、写 N 次。
//   访存工作量相差 O(N) 倍，实际加速比还受 cache、exp 和并行度影响。
// 问答题 07.3 答案：exp(m1−m) 就是旧分块的 rescale 因子。FlashAttention 用它
//   同时缩放旧 denominator 和输出累加量，再加入新分块。
