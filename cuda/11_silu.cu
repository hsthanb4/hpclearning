#include <cuda_runtime.h>

// SiLU / Swish Activation:
// y = x * sigmoid(x) = x / (1 + exp(-x))
//
// 在 LLM 里常见的 SwiGLU 会用到 SiLU：
// out = silu(gate) * up
//
// 这里写单独 SiLU 算子，面试时可以讲 elementwise kernel 的基本优化点：
// 1. grid-stride loop 处理任意长度。
// 2. 相邻线程访问相邻元素，global memory load/store 合并。
// 3. 计算主要瓶颈是 expf，进一步优化可以讨论近似 sigmoid 或融合到上游 GEMM。
__global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
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
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid = grid > 4096 ? 4096 : grid;
    silu_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(input, output, n);
}
