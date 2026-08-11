#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

// ArgMax: 找到 input[0:n] 中最大值的索引。
//
// 面试重点：
// 1. reduce 时同时传递 (value, index) 对。
// 2. warp shuffle 传 value 比较，再传 index 选择。
// 3. 两轮 kernel：第一轮每个 block 出一个 (val, idx)，第二轮归并。

struct ValIdx {
    float val;
    int idx;
};

template<int BLOCK_SIZE>
__device__ __forceinline__ ValIdx warp_reduce_max(ValIdx vi) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, vi.val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, vi.idx, offset);
        if (other_val > vi.val) {
            vi.val = other_val;
            vi.idx = other_idx;
        }
    }
    return vi;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ ValIdx block_reduce_max(ValIdx vi) {
    constexpr int NUM_WARPS = (BLOCK_SIZE + 31) / 32;
    __shared__ float smem_val[NUM_WARPS];
    __shared__ int smem_idx[NUM_WARPS];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    vi = warp_reduce_max<BLOCK_SIZE>(vi);

    if (lane == 0) {
        smem_val[warp] = vi.val;
        smem_idx[warp] = vi.idx;
    }
    __syncthreads();

    if (warp == 0) {
        vi.val = (lane < NUM_WARPS) ? smem_val[lane] : -FLT_MAX;
        vi.idx = (lane < NUM_WARPS) ? smem_idx[lane] : -1;
        vi = warp_reduce_max<BLOCK_SIZE>(vi);
    }

    return vi;
}

// 第一轮：每个 block 用 grid-stride loop 找到局部最大值及其索引
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

    // 每个线程先通过 grid-stride loop 找到自己负责的元素中的最大值
    ValIdx local = {-FLT_MAX, -1};
    for (int i = idx; i < n; i += stride) {
        if (input[i] > local.val) {
            local.val = input[i];
            local.idx = i;
        }
    }

    // block 级 reduce
    ValIdx result = block_reduce_max<BLOCK_SIZE>(local);

    if (tid == 0) {
        partial_val[blockIdx.x] = result.val;
        partial_idx[blockIdx.x] = result.idx;
    }
}

// 第二轮：把所有 block 的 partial 结果归并，只需一个 block
template<int BLOCK_SIZE>
__global__ void argmax_final_kernel(
    const float* __restrict__ partial_val,
    const int* __restrict__ partial_idx,
    int* __restrict__ output,
    int num_blocks
) {
    int tid = threadIdx.x;

    ValIdx local = {-FLT_MAX, -1};
    // 一个 block 处理所有 partial，stride = BLOCK_SIZE
    for (int i = tid; i < num_blocks; i += BLOCK_SIZE) {
        if (partial_val[i] > local.val) {
            local.val = partial_val[i];
            local.idx = partial_idx[i];
        }
    }

    ValIdx result = block_reduce_max<BLOCK_SIZE>(local);

    if (tid == 0) {
        *output = result.idx;
    }
}

} // namespace

// ===================== Host 调用接口 =====================

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
    argmax_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_input, d_partial_val, d_partial_idx, n
    );
    argmax_final_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        d_partial_val, d_partial_idx, d_output, num_blocks
    );
}

// ===================== 测试 =====================

int main() {
    const int N = 1 << 20;  // 1M 元素
    const int NUM_BLOCKS = 128;

    // Host 数据
    float* h_input = (float*)malloc(N * sizeof(float));
    srand(42);
    int expected_idx = 0;
    float expected_val = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
        if (h_input[i] > expected_val) {
            expected_val = h_input[i];
            expected_idx = i;
        }
    }

    // Device 数据
    float *d_input, *d_partial_val;
    int *d_partial_idx, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_partial_val, NUM_BLOCKS * sizeof(float));
    cudaMalloc(&d_partial_idx, NUM_BLOCKS * sizeof(int));
    cudaMalloc(&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 执行
    launch_argmax(d_input, d_output, d_partial_val, d_partial_idx, N, NUM_BLOCKS, 0);

    // 取结果
    int h_output;
    cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("ArgMax result: index = %d, value = %f\n", h_output, h_input[h_output]);
    printf("Expected:      index = %d, value = %f\n", expected_idx, expected_val);
    printf("%s\n", (h_output == expected_idx) ? "PASS" : "FAIL");

    // 清理
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_partial_val);
    cudaFree(d_partial_idx);
    cudaFree(d_output);

    return 0;
}
