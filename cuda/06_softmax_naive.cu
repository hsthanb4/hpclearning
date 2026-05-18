#include <float.h>
#include <cuda_runtime.h>

// Naive Softmax: 对 [M, N] 的每一行做 softmax。
//
// 这个版本保留“一个线程负责一个输出元素”的写法，
// 适合面试里先讲清楚 softmax 的数学公式和它为什么低效。
//
// output[row, col] = exp(input[row, col] - max(row)) / sum(exp(input[row, i) - max(row)))
//
// 低效点：
// 对同一行来说，max(row) 和 sum(row) 是所有 col 共享的，
// 但这里每个输出元素线程都会重新扫完整行求 max 和 sum。
// 所以每行复杂度接近 O(N^2)，优化版应改成一个 block 合作处理一行。
__global__ void softmax_naive_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float max_val = -FLT_MAX;

        // 第一次扫整行，求最大值，避免 exp 溢出。
        for (int i = 0; i < N; ++i) {
            max_val = fmaxf(max_val, input[row * N + i]);
        }

        float sum_exp = 0.0f;

        // 第二次扫整行，求 denominator。
        for (int i = 0; i < N; ++i) {
            sum_exp += expf(input[row * N + i] - max_val);
        }

        // 当前线程只写自己负责的一个 output[row, col]。
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
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    softmax_naive_kernel<<<grid, block, 0, stream>>>(input, output, M, N);
}
