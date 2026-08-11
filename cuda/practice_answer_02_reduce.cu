// ============================================================================
// practice_answer_02_reduce.cu
// 对应练习文件：practice_02_reduce.cu（先做填空，再对照本答案）
// 各练习使用独立 namespace（p02/p03/...）避免名称冲突。
// ============================================================================

#include <cuda_runtime.h>
#include <float.h>

// ----------------------------------------------------------------------------
// 练习 02：Reduce Sum —— 完整答案
// ----------------------------------------------------------------------------
namespace p02 {

template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // TODO 1: shuffle 归约，offset 16->8->4->2->1
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024 && BLOCK_SIZE % 32 == 0,
                  "BLOCK_SIZE must contain full warps");
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    // TODO 2: 先 warp reduce（每 warp 出一个数）
    val = warp_reduce_sum<BLOCK_SIZE>(val);

    // TODO 3: 每 warp 的 lane 0 写 shared memory
    if (lane == 0) {
        smem[warp] = val;
    }
    __syncthreads();

    // TODO 4: warp 0 读回所有 partial，再 warp reduce 一次
    if (warp == 0) {
        val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }
    return val;
}

template<int BLOCK_SIZE>
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial,
    int n
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // TODO 5: 每线程先寄存器累加多个元素
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        local_sum += input[i];
    }

    float block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    // TODO 6: 每 block 只写一个 partial
    if (tid == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

} // namespace p02

void launch_reduce_sum_stage(
    const float* input,
    float* partial,
    int n,
    int num_blocks,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    if (num_blocks <= 0) return;

    // TODO 7: launch，grid=num_blocks
    p02::reduce_sum_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, partial, n
    );
}

// 问答题 02.1 答案：0xffffffff 表示 32 个 lane 都参与。掩码中的线程必须执行
//   同一次 shuffle。本例固定启动 256 线程，所以全掩码安全；发散代码需使用
//   与实际参与 lane 一致的掩码。输入尾部不会改变 block 的线程数。
// 问答题 02.2 答案：普通 block 间无法在 kernel 内全局同步。两阶段先生成
//   partial，再用第二次 launch 归并，避免全局原子竞争。代价是额外 launch；
//   若每 block 只 atomicAdd 一次，小规模时也可能更快，但求和顺序不确定。
// 问答题 02.3 答案：warp 内先用 shuffle 归约，每个 warp 只向 shared memory
//   写一个 partial，因而减少 shared memory 用量、访问和同步开销。

// ----------------------------------------------------------------------------
// 练习 03：GEMV —— 完整答案
// ----------------------------------------------------------------------------
namespace p03 {

template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024 && BLOCK_SIZE % 32 == 0,
                  "BLOCK_SIZE must contain full warps");
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_sum<BLOCK_SIZE>(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }
    return val;
}

template<int BLOCK_SIZE>
__global__ void gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ y,
    int M,
    int N
) {
    // TODO 9: 一个 block 负责一行
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    const float* row_A = A + row * N;
    float local_sum = 0.0f;

    // TODO 10: 行内 stride 点积
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        local_sum += row_A[col] * x[col];
    }

    float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    // TODO 11: 单线程写回（bias 可空）
    if (tid == 0) {
        y[row] = sum + (bias ? bias[row] : 0.0f);
    }
}

} // namespace p03

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
    if (M <= 0) return;

    // TODO 12: grid=M（每行一个 block）
    p03::gemv_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        A, x, bias, y, M, N
    );
}

// 问答题 03.1 答案：每行都需读完整 x。可让一个 block 处理多行，把 x 分块
//   读入 shared memory 共享；多向量时则改用 GEMM 以获得更多复用。
// 问答题 03.2 答案：对较小的 x，重复读通常命中 cache；A 则只使用一次，
//   是主要流式读取。若 x 无法驻留 cache，这一结论不成立。
// 问答题 03.3 答案：首轮只有 N 个线程有数据，但仍要支付 256 线程的归约成本。
//   小 N 常用每 warp 一行，让一个 block 处理多行；最佳方案还取决于 M。

// ----------------------------------------------------------------------------
// 练习 04：RMSNorm —— 完整答案
// ----------------------------------------------------------------------------
namespace p04 {

template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024 && BLOCK_SIZE % 32 == 0,
                  "BLOCK_SIZE must contain full warps");
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_sum<BLOCK_SIZE>(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }
    return val;
}

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

    // TODO 14: pass1 累加 x*x（只需要平方和）
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];
        local_sq_sum += x * x;
    }

    float sq_sum = block_reduce_sum<BLOCK_SIZE>(local_sq_sum);

    // TODO 15: tid0 算 inv_rms，smem 广播
    __shared__ float smem_inv_rms;
    if (tid == 0) {
        smem_inv_rms = rsqrtf(sq_sum / static_cast<float>(N) + eps);
    }
    __syncthreads();

    float inv_rms = smem_inv_rms;

    // TODO 16: pass2 归一化
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        row_out[col] = row_in[col] * inv_rms * weight[col];
    }
}

} // namespace p04

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
    if (M <= 0 || N <= 0) return;

    p04::rmsnorm_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        input, weight, output, M, N, eps
    );
}

// 问答题 04.1 答案：RMSNorm 不求 mean，也不做中心化和 beta 偏移，因而统计量
//   和算术更少。它在许多 LLM 中效果足够且成本更低，但不是所有模型都必然更优。
// 问答题 04.2 答案：block reduce 的完整 sq_sum 只在 tid 0 有效。tid 0 计算一次
//   inv_rms，再通过 shared memory 和 __syncthreads 广播给全 block。
// 问答题 04.3 答案：输出前必须先知道整行平方和。只有把 x 保留在寄存器或
//   shared memory 中，才能只读一次 global memory；N 大到放不下时，仍需二次全局读或额外存储。

// ----------------------------------------------------------------------------
// 练习 05：LayerNorm —— 完整答案
// ----------------------------------------------------------------------------
namespace p05 {

template<int BLOCK_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024 && BLOCK_SIZE % 32 == 0,
                  "BLOCK_SIZE must contain full warps");
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    val = warp_reduce_sum<BLOCK_SIZE>(val);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (lane < NUM_WARPS) ? smem[lane] : 0.0f;
        val = warp_reduce_sum<BLOCK_SIZE>(val);
    }
    __syncthreads();
    return val;
}

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

    // TODO 19: pass1 同时累加 sum 与 sq_sum
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float x = row_in[col];
        local_sum += x;
        local_sq_sum += x * x;
    }

    // 两个独立 reduce（连续调用 block_reduce_sum 是安全的，
    // 因为函数末尾有 __syncthreads 兜底）
    float sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
    float sq_sum = block_reduce_sum<BLOCK_SIZE>(local_sq_sum);

    __shared__ float smem_mean;
    __shared__ float smem_inv_std;

    if (tid == 0) {
        // TODO 20: mean/var 合成，fmaxf 防负
        float mean = sum / static_cast<float>(N);
        float var = sq_sum / static_cast<float>(N) - mean * mean;
        smem_mean = mean;
        smem_inv_std = rsqrtf(fmaxf(var, 0.0f) + eps);
    }
    __syncthreads();

    float mean = smem_mean;
    float inv_std = smem_inv_std;

    // TODO 21: pass2 归一化 + affine
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float y = (row_in[col] - mean) * inv_std;
        row_out[col] = y * gamma[col] + beta[col];
    }
}

} // namespace p05

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
    if (M <= 0 || N <= 0) return;

    p05::layernorm_kernel<BLOCK_SIZE><<<M, BLOCK_SIZE, 0, stream>>>(
        input, gamma, beta, output, M, N, eps
    );
}

// 问答题 05.1 答案：E[x²]−E[x]² 会因大数相减丢失精度，甚至得到负方差。
//   更稳定的方法是先求 mean 再求 Σ(x−mean)²，或使用 Welford 算法。fmaxf 只是兜底。
// 问答题 05.2 答案：函数末尾的 __syncthreads() 确保第一次归约已读完 smem，
//   第二次调用才能覆盖它，避免读写竞争。
// 问答题 05.3 答案：寄存器是线程私有的。tid 0 将 mean 和 inv_std 写入 shared
//   memory，同步后才能向全 block 广播。

// ----------------------------------------------------------------------------
// 练习 12：ArgMax —— 完整答案
// ----------------------------------------------------------------------------
namespace p12 {

struct ValIdx {
    float val;
    int idx;
};

__device__ __forceinline__ bool better(ValIdx candidate, ValIdx current) {
    return candidate.idx >= 0 &&
           (current.idx < 0 || candidate.val > current.val ||
            (candidate.val == current.val && candidate.idx < current.idx));
}

template<int BLOCK_SIZE>
__device__ __forceinline__ ValIdx warp_reduce_max(ValIdx vi) {
    // TODO 23/24: 同时 shuffle val 和 idx，比较后同步替换
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        ValIdx other = {
            __shfl_down_sync(0xffffffff, vi.val, offset),
            __shfl_down_sync(0xffffffff, vi.idx, offset)
        };
        if (better(other, vi)) vi = other;
    }
    return vi;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ ValIdx block_reduce_max(ValIdx vi) {
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024 && BLOCK_SIZE % 32 == 0,
                  "BLOCK_SIZE must contain full warps");
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem_val[NUM_WARPS];
    __shared__ int smem_idx[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    // TODO 25: warp reduce 后 lane0 写 smem（val 和 idx 都要写）
    vi = warp_reduce_max<BLOCK_SIZE>(vi);
    if (lane == 0) {
        smem_val[warp] = vi.val;
        smem_idx[warp] = vi.idx;
    }
    __syncthreads();

    // TODO 26: warp0 读回，空位用 -FLT_MAX/-1 填充，再 reduce
    if (warp == 0) {
        vi.val = (lane < NUM_WARPS) ? smem_val[lane] : -FLT_MAX;
        vi.idx = (lane < NUM_WARPS) ? smem_idx[lane] : -1;
        vi = warp_reduce_max<BLOCK_SIZE>(vi);
    }
    return vi;
}

template<int BLOCK_SIZE>
__global__ void argmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ partial_val,
    int* __restrict__ partial_idx,
    int n
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    ValIdx local = {-FLT_MAX, -1};

    // TODO 27: grid-stride 找局部最大（val 和 idx 同步更新）
    for (int i = idx; i < n; i += stride) {
        ValIdx candidate = {input[i], i};
        if (better(candidate, local)) local = candidate;
    }

    ValIdx result = block_reduce_max<BLOCK_SIZE>(local);

    // TODO 28: 每 block 出一个 (val, idx)
    if (tid == 0) {
        partial_val[blockIdx.x] = result.val;
        partial_idx[blockIdx.x] = result.idx;
    }
}

template<int BLOCK_SIZE>
__global__ void argmax_final_kernel(
    const float* __restrict__ partial_val,
    const int* __restrict__ partial_idx,
    int* __restrict__ output,
    int num_blocks
) {
    int tid = threadIdx.x;

    ValIdx local = {-FLT_MAX, -1};

    // TODO 29: 单 block 扫所有 partial
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
        ValIdx candidate = {partial_val[i], partial_idx[i]};
        if (better(candidate, local)) local = candidate;
    }

    ValIdx result = block_reduce_max<BLOCK_SIZE>(local);
    if (tid == 0) {
        *output = result.idx;
    }
}

} // namespace p12

void launch_argmax(
    const float* d_input,
    int* d_output,
    float* d_partial_val,
    int* d_partial_idx,
    int n,
    int num_blocks,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    if (n <= 0 || num_blocks <= 0) return;

    // TODO 30: 第一轮 num_blocks 个 block，第二轮 1 个 block 归并
    p12::argmax_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_input, d_partial_val, d_partial_idx, n
    );
    p12::argmax_final_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        d_partial_val, d_partial_idx, d_output, num_blocks
    );
}

// 问答题 12.1 答案：单 block 只能占用一个 SM，无法利用整张 GPU。两阶段先并行
//   生成局部最大值，再归并少量 partial，提高并行度。
// 问答题 12.2 答案：比较值时必须同步携带它的索引，否则归约后无法知道
//   最大值来自哪个元素。分开归约会需要额外的 winner 跟踪。
// 问答题 12.3 答案：单纯使用 > 不能保证全局最小索引，结果取决于归约树。
//   本实现在值相等时显式选更小 idx，因而对无 NaN 输入稳定返回第一个最大值。
