#include <cuda_runtime.h>

namespace {

// Reduce Sum: 把 input[0:n] 求和。
//
// 面试重点：
// 1. 每个线程先在寄存器里累加多个元素，减少后续 reduce 的数据量。
// 2. warp 内用 shuffle reduce。
// 3. 跨 warp 用 shared memory 保存每个 warp 的 partial sum。
//
// 这个文件实现的是单 stage reduction：
// 每个 block 输出一个 partial sum 到 partial[blockIdx.x]。
// 如果 partial 还有多个元素，可以继续对 partial 再 launch 本 kernel。
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
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    val = warp_reduce_sum<BLOCK_SIZE>(val);

    if (lane == 0) {
        smem[warp] = val;
    }

    __syncthreads();

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

    float local_sum = 0.0f;

    // grid-stride loop：每个线程累加多个元素。
    // 比“一线程一个元素然后立刻 reduce”更高效，因为寄存器累加很便宜。
    for (int i = idx; i < n; i += stride) {
        local_sum += input[i];
    }

    float block_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);

    // 一个 block 只写一个 partial sum，后续由下一轮 kernel 再归并。
    if (tid == 0) {
        partial[blockIdx.x] = block_sum;
    }
}

} // namespace

void launch_reduce_sum_stage(
    const float* input,
    float* partial,
    int n,
    int num_blocks,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    reduce_sum_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, partial, n
    );
}
