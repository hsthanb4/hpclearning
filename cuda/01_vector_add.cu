#include <cuda_runtime.h>

// Vector Add: c[i] = a[i] + b[i]
//
// 这是 CUDA 面试里最基础的算子，重点不是算法复杂，
// 而是确认你理解 thread/block/grid 映射和 coalesced memory access。
__global__ void vector_add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // grid-stride loop 的好处：
    // 1. kernel 可以处理任意 n，不要求 grid 覆盖全部元素后就停止。
    // 2. 如果 n 很大，每个线程可以处理多个元素，代码仍然简单。
    // 3. 相邻线程访问相邻 idx，读写都是合并访存。
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
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 限制 grid 上限是常见写法，避免 n 极大时创建过多 block。
    // grid-stride loop 会保证剩余元素继续被处理。
    grid = grid > 4096 ? 4096 : grid;

    vector_add_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(a, b, c, n);
}



