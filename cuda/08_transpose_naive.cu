#include <cuda_runtime.h>

// Naive Transpose:
// input:  [M, N] row-major
// output: [N, M] row-major
//
// 这个版本一个线程搬一个元素。
// 读 input 时，相邻线程读 input[row, col] 的连续地址，读是合并的；
// 写 output[col, row] 时，相邻线程写的地址跨度是 M，写通常不合并。
//
// 面试讲法：
// naive transpose 的瓶颈通常在 strided global store，
// 优化版使用 shared memory tile，把 strided store 变成 coalesced store。
__global__ void transpose_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

void launch_transpose_naive(
    const float* input,
    float* output,
    int M,
    int N,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    transpose_naive_kernel<<<grid, block, 0, stream>>>(input, output, M, N);
}
