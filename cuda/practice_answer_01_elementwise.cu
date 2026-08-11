// ============================================================================
// practice_answer_01_elementwise.cu
// 对应练习文件：practice_01_elementwise.cu（先做填空，再对照本答案）
// ============================================================================

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// 练习 01：Vector Add —— 完整答案
// ----------------------------------------------------------------------------
__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    // TODO 1: 全局下标
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO 2: stride
    int stride = blockDim.x * gridDim.x;
    // TODO 3: grid-stride loop
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

void launch_vector_add(
    const float* a,
    const float* b,
    float* c,
    int n,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    if (n <= 0) return;

    // TODO 4: 向上取整 + 上限 4096
    int grid = (n - 1) / BLOCK_SIZE + 1;
    grid = grid > 4096 ? 4096 : grid;
    // TODO 5: 4 参 launch
    vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
}

// 问答题 01.1 答案：线程可复用，grid 无需覆盖全部数据；因此能限制
//   block 数，同一 kernel 仍可处理任意 n。
// 问答题 01.2 答案：warp 访问对齐的 32 个连续 float 时，通常合并为 4 个 32B 事务。
//   若 idx = threadIdx.x，所有 block 会重复读写同一批元素，写入还会产生数据竞争。
// 问答题 01.3 答案：grid=391；最后一个 block 有 160 个有效线程，96 个线程空转。

// ----------------------------------------------------------------------------
// 练习 11：SiLU —— 完整答案
// ----------------------------------------------------------------------------
__global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    // TODO 6/7: idx / stride / loop / 公式
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void launch_silu(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    if (n <= 0) return;

    int grid = (n - 1) / BLOCK_SIZE + 1;
    grid = grid > 4096 ? 4096 : grid;
    silu_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n);
}

// 问答题 11.1 答案：普通 elementwise 常受带宽限制，但 SiLU 也可能受 expf
//   吞吐限制。用 profiler 查显存带宽和特殊函数单元利用率，再与纯拷贝对比。
// 问答题 11.2 答案：可用 __expf、多项式或查表近似；向量化可减少指令开销；
//   更有效的做法是与相邻算子或 GEMM epilogue 融合。近似方法需先验证误差。
// 问答题 11.3 答案：融合后 gate 可直接与 up 相乘，避免 SiLU 中间张量的
//   一次全局写和一次全局读，并减少 kernel launch。
