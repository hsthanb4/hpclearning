# Day 1: GPU 架构与 CUDA 基础

**学习日期**: 2026-02-02
**学习目标**: 理解 GPU 硬件架构和 CUDA 编程模型
**预计学习时间**: 8小时 (上午3h + 下午3h + 晚上2h)

---

## 📋 学习大纲

### 上午 (3小时)
- GPU 架构基础
- 阅读 NVIDIA CUDA C Programming Guide (Chapters 1-3)

### 下午 (3小时)
- CUDA 编程入门
- 实践: 编写并运行 5 个基础 CUDA 程序

### 晚上 (2小时)
- 性能分析工具
- 作业: 实现并优化矩阵乘法 (naive version)

---

## 📚 第一部分: GPU 架构基础 (上午)

### 1.1 GPU vs CPU 架构对比

#### CPU 特点
- **设计理念**: 为通用计算设计，强调单线程性能
- **核心数量**: 少量核心 (8-64核)
- **缓存**: 大容量多级缓存 (L1/L2/L3)
- **控制逻辑**: 复杂的分支预测、乱序执行
- **适用场景**: 串行任务、复杂逻辑、低延迟计算

#### GPU 特点
- **设计理念**: 为大规模并行计算设计，强调吞吐量
- **核心数量**: 大量简单核心 (数千到上万个)
- **缓存**: 小容量缓存，更多片上共享内存
- **控制逻辑**: 简单的流水线，统一的指令执行
- **适用场景**: 数据并行任务、矩阵运算、图形渲染

#### 为什么 GPU 适合深度学习？
1. **矩阵运算**: 神经网络的核心是矩阵乘法，天然适合并行化
2. **数据并行**: 同一操作应用到大量数据上 (SIMD - Single Instruction Multiple Data)
3. **内存带宽**: 高带宽内存 (HBM) 支持快速数据传输
4. **专用硬件**: Tensor Core 等专用单元加速特定运算

---

### 1.2 GPU 硬件架构详解

#### 1.2.1 GPU 层次结构

```
GPU Device
│
├── GPC (Graphics Processing Cluster) × N
│   │
│   ├── SM (Streaming Multiprocessor) × M
│   │   │
│   │   ├── CUDA Core × 64-128
│   │   ├── Tensor Core × 4-8 (Volta+)
│   │   ├── Special Function Units (SFU)
│   │   ├── Load/Store Units (LD/ST)
│   │   ├── Warp Scheduler × 4
│   │   ├── Dispatch Unit × 8
│   │   ├── Register File (256KB - 65536 registers)
│   │   ├── Shared Memory (L1 Cache) - 64KB-128KB
│   │   └── L1 Cache / Texture Cache
│   │
│   └── L2 Cache (共享)
│
├── Global Memory (GDDR6/HBM2/HBM3)
└── Memory Controllers
```

#### 1.2.2 SM (Streaming Multiprocessor) - GPU 的核心单元

SM 是 GPU 的基本执行单元，包含:

**计算单元**:
- **CUDA Cores**: 执行浮点和整数运算 (FP32, INT32)
  - 每个 SM 有 64-128 个 CUDA Cores
- **Tensor Cores** (Volta/Turing/Ampere/Hopper):
  - 专门用于矩阵乘累加 (Matrix Multiply-Accumulate)
  - 支持 FP16, BF16, TF32, FP8, INT8, INT4
  - 单个 Tensor Core 可在一个时钟周期完成 4×4×4 矩阵乘法
- **Special Function Units (SFU)**:
  - 快速计算超越函数 (sin, cos, log, exp, sqrt)
  - 精度略低但速度快

**控制单元**:
- **Warp Scheduler**: 调度 Warp (32个线程为一组) 执行
- **Dispatch Unit**: 分发指令到执行单元

**内存单元**:
- **Register File**: 每个 SM 有 64KB - 256KB 寄存器
  - 访问速度最快 (1 cycle)
  - 每个线程独享部分寄存器
- **Shared Memory**: 64KB - 128KB 可编程共享内存
  - 在一个 Block 内的所有线程共享
  - 低延迟 (~20-30 cycles)
  - 可配置为 L1 Cache
- **L1 Cache / Texture Cache**: 硬件管理的缓存

---

#### 1.2.3 Warp 调度机制

**什么是 Warp？**
- Warp 是 GPU 调度和执行的基本单位
- 1 个 Warp = 32 个线程
- 同一个 Warp 中的 32 个线程执行相同的指令 (SIMT - Single Instruction Multiple Thread)

**Warp 调度**:
```
Block (例如 256 threads)
│
├── Warp 0 (Thread 0-31)
├── Warp 1 (Thread 32-63)
├── Warp 2 (Thread 64-95)
└── ...
└── Warp 7 (Thread 224-255)
```

**调度策略**:
1. SM 可以驻留多个 Block
2. 每个时钟周期，Warp Scheduler 选择一个就绪的 Warp 执行
3. 当一个 Warp 等待内存访问时，调度器切换到其他 Warp (隐藏延迟)

**Warp Divergence (分支发散)**:
```cuda
// 问题代码: Warp 内线程执行不同路径
if (threadIdx.x < 16) {
    // 前16个线程执行这里
    result = a + b;
} else {
    // 后16个线程执行这里
    result = a * b;
}
// 实际执行: 两个分支都会执行，不满足条件的线程被 mask 掉
// 性能降低 50%
```

**避免 Warp Divergence**:
- 尽量保证同一个 Warp 内的线程执行相同的代码路径
- 使用 `__ballot_sync()`, `__any_sync()`, `__all_sync()` 等 warp-level 原语

---

#### 1.2.4 内存层次结构

GPU 内存层次 (从快到慢):

| 内存类型 | 大小 | 延迟 | 带宽 | 作用域 | 生命周期 |
|---------|------|------|------|--------|---------|
| **Register** | 256KB/SM | 1 cycle | 最高 | 单个线程 | 线程 |
| **Shared Memory** | 64-128KB/SM | 20-30 cycles | ~8TB/s | Block 内所有线程 | Block |
| **L1 Cache** | 64-128KB/SM | 20-30 cycles | 自动管理 | Block | 自动 |
| **L2 Cache** | 6-60MB | 100-200 cycles | ~4TB/s | 所有 SM | 自动 |
| **Global Memory** | 16-80GB | 400-800 cycles | 1-2TB/s | 所有线程 | 应用程序 |
| **CPU Memory** | GB-TB | 数万 cycles | ~100GB/s | CPU+GPU | 手动管理 |

**内存访问优化原则**:
1. **寄存器优先**: 尽量使用局部变量
2. **Shared Memory 次之**: 用于线程间协作
3. **避免 Global Memory**: 必要时使用合并访问
4. **内存带宽优化**: 提高计算/访存比 (Compute to Memory Ratio)

---

#### 1.2.5 Tensor Core 原理

**Tensor Core 架构** (以 Ampere A100 为例):

```
输入: A (FP16/BF16) × B (FP16/BF16)
累加: C (FP32)
输出: D (FP32) = A × B + C

一个 Tensor Core 在一个时钟周期完成:
D[4×4] = A[4×4] × B[4×4] + C[4×4]
```

**支持的精度**:
- **Volta**: FP16
- **Turing**: FP16, INT8, INT4, INT1
- **Ampere**: FP16, BF16, TF32 (自动), FP64 (A100), INT8, INT4
- **Hopper**: FP8, FP16, BF16, TF32, FP64, INT8

**TF32 (TensorFloat-32)**:
- Mantissa: 10 bits (FP16 级别)
- Exponent: 8 bits (FP32 级别)
- 自动替换 FP32 的 GEMM，无需代码修改
- 性能提升 8x，精度损失小于 0.1%

**使用 Tensor Core 的方式**:
1. **WMMA API** (Warp Matrix Multiply-Accumulate)
2. **cuBLAS / cuDNN**
3. **PyTorch / TensorFlow** (自动使用)

---

### 1.3 主流 GPU 架构对比

#### 1.3.1 架构代号与产品对应

| 架构 | 代号 | Compute Capability | 代表产品 | 发布年份 |
|-----|------|-------------------|---------|---------|
| Kepler | GK110 | 3.5, 3.7 | K40, K80 | 2012 |
| Maxwell | GM200 | 5.0, 5.2 | GTX 980, Titan X | 2014 |
| Pascal | GP100 | 6.0, 6.1 | P100, GTX 1080 | 2016 |
| Volta | GV100 | 7.0 | V100, Titan V | 2017 |
| Turing | TU102 | 7.5 | RTX 2080, T4 | 2018 |
| Ampere | GA102/100 | 8.0, 8.6, 8.7 | A100, RTX 3090, RTX 3060 | 2020 |
| Ada Lovelace | AD102 | 8.9 | RTX 4090, RTX 4060 | 2022 |
| Hopper | GH100 | 9.0 | H100 | 2022 |
| Blackwell | GB100 | 10.0 | B100, Thor | 2024 |

#### 1.3.2 关键技术演进

**Volta (2017) - 深度学习的转折点**:
- 首次引入 **Tensor Core**
- 独立的线程调度 (Independent Thread Scheduling)
- NVLink 2.0 (300GB/s)

**Ampere (2020) - AI 训练主力**:
- 第三代 Tensor Core (支持 TF32, BF16, FP64)
- Sparse Tensor Core (2:4 结构化稀疏, 2x 加速)
- MIG (Multi-Instance GPU) - 虚拟化

**Hopper (2022) - 最新旗舰**:
- 第四代 Tensor Core (FP8 支持)
- Transformer Engine (自动 FP8 转换)
- Thread Block Clusters (跨 SM 协作)
- DPX Instructions (动态规划加速)
- NVLink 4.0 (900GB/s)

---

### 1.4 本仓库支持的 GPU 架构

根据 `CLAUDE.md` 中的配置，当前仓库支持的 GPU:

| GPU 型号 | 架构 | SM 版本 | CMake 配置 |
|---------|-----|---------|-----------|
| A100 | Ampere | `sm_80` | `-DCMAKE_CUDA_ARCHITECTURES=80` |
| RTX 3060 | Ampere | `sm_86` | `-DCMAKE_CUDA_ARCHITECTURES=86` |
| Orin | Ampere | `sm_87` | `-DCMAKE_CUDA_ARCHITECTURES=87` |
| RTX 4060 | Ada Lovelace | `sm_89` | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| Thor | Blackwell | `sm_100` | `-DCMAKE_CUDA_ARCHITECTURES=100` |

**自动检测当前 GPU**:
```bash
# CMake 会自动检测
cmake -DCMAKE_CUDA_ARCHITECTURES=native ..
```

---

## 💻 第二部分: CUDA 编程入门 (下午)

### 2.1 CUDA 编程模型

#### 2.1.1 Host 和 Device

- **Host**: CPU 及其内存
- **Device**: GPU 及其内存

**典型 CUDA 程序流程**:
```
1. Host 分配内存
2. Host 将数据复制到 Device (cudaMemcpy H2D)
3. Host 启动 Kernel (<<<grid, block>>>)
4. Device 执行 Kernel
5. Host 将结果复制回 Host (cudaMemcpy D2H)
6. Host 释放内存
```

---

#### 2.1.2 Thread / Block / Grid 层次结构

```
Grid (Kernel Launch)
│
├── Block (0,0)   Block (1,0)   Block (2,0)
│   │
│   ├── Thread (0,0)  Thread (1,0)  ...  Thread (15,0)
│   ├── Thread (0,1)  Thread (1,1)  ...  Thread (15,1)
│   └── ...
│
├── Block (0,1)   Block (1,1)   Block (2,1)
└── ...
```

**术语**:
- **Thread**: 最小执行单元
- **Block**: 线程组，共享 Shared Memory
- **Grid**: Block 的集合，一个 Kernel 对应一个 Grid

**内置变量**:
```cuda
// Thread 索引
threadIdx.x, threadIdx.y, threadIdx.z  // Block 内的线程索引
blockIdx.x, blockIdx.y, blockIdx.z     // Grid 内的 Block 索引
blockDim.x, blockDim.y, blockDim.z     // Block 的维度
gridDim.x, gridDim.y, gridDim.z        // Grid 的维度

// 计算全局线程索引
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

---

### 2.2 CUDA 程序实例

#### 2.2.1 示例 1: Hello World

```cuda
// hello.cu
#include <stdio.h>

// __global__ 表示这是一个 kernel 函数
__global__ void helloKernel() {
    printf("Hello from thread %d in block %d\n",
           threadIdx.x, blockIdx.x);
}

int main() {
    // 启动 2 个 Block，每个 Block 有 4 个 Thread
    helloKernel<<<2, 4>>>();

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    return 0;
}
```

**编译和运行**:
```bash
nvcc hello.cu -o hello
./hello
```

**输出**:
```
Hello from thread 0 in block 0
Hello from thread 1 in block 0
Hello from thread 2 in block 0
Hello from thread 3 in block 0
Hello from thread 0 in block 1
Hello from thread 1 in block 1
Hello from thread 2 in block 1
Hello from thread 3 in block 1
```

---

#### 2.2.2 示例 2: 向量加法

```cuda
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: 向量加法
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 20;  // 1M 元素
    size_t bytes = N * sizeof(float);

    // 1. 分配 Host 内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. 复制数据到 Device (Host to Device)
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. 启动 Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 5. 复制结果回 Host (Device to Host)
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 6. 验证结果
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %.2f\n", i, h_C[i]);
    }

    // 7. 释放内存
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

**关键点**:
- `cudaMalloc()`: 在 GPU 上分配内存
- `cudaMemcpy()`: 在 CPU 和 GPU 之间复制数据
- `<<<blocksPerGrid, threadsPerBlock>>>`: Kernel 启动配置
- `if (i < N)`: 边界检查 (因为线程数可能超过数组大小)

---

#### 2.2.3 示例 3: 矩阵加法

```cuda
// matrix_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 1024
#define HEIGHT 1024

// 2D Kernel: 矩阵加法
__global__ void matrixAdd(const float *A, const float *B, float *C,
                          int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    size_t bytes = WIDTH * HEIGHT * sizeof(float);

    // 分配和初始化 Host 内存
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 2D Grid 和 Block 配置
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 blocksPerGrid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, WIDTH, HEIGHT);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 验证
    printf("C[0] = %.2f (expected 3.00)\n", h_C[0]);
    printf("C[%d] = %.2f (expected 3.00)\n", WIDTH*HEIGHT-1, h_C[WIDTH*HEIGHT-1]);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

**2D 配置优势**:
- 自然映射到 2D 矩阵
- 更好的内存访问模式
- 代码更易读

---

#### 2.2.4 示例 4: Shared Memory 使用 - 矩阵转置

```cuda
// transpose.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define WIDTH 1024

// 使用 Shared Memory 避免 Bank Conflict
__global__ void transposeShared(const float *input, float *output,
                                int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 避免 bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // 从 Global Memory 读取到 Shared Memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();  // 确保所有线程都完成读取

    // 转置后的坐标
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 从 Shared Memory 写回 Global Memory (转置)
    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    size_t bytes = WIDTH * WIDTH * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // 初始化
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_input[i] = i;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 blocksPerGrid((WIDTH + TILE_DIM - 1) / TILE_DIM,
                       (WIDTH + TILE_DIM - 1) / TILE_DIM);

    transposeShared<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, WIDTH, WIDTH);

    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // 验证
    printf("input[1][0] = %.2f, output[0][1] = %.2f\n",
           h_input[1 * WIDTH + 0], h_output[0 * WIDTH + 1]);

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);

    return 0;
}
```

**关键技术**:
- `__shared__`: 声明 Shared Memory
- `__syncthreads()`: Block 内线程同步
- `[TILE_DIM][TILE_DIM + 1]`: 避免 Bank Conflict

---

#### 2.2.5 示例 5: 规约 (Reduction) - 求和

```cuda
// reduction.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// Reduction Kernel: Block 内规约
__global__ void reductionSum(const float *input, float *output, int N) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到 Shared Memory
    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    // 树形规约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回结果 (每个 Block 一个结果)
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1 << 20;  // 1M
    size_t bytes = N * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // 所有元素为 1，总和应为 N
    }

    float *d_input, *d_output;
    int blocksPerGrid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // 第一次规约
    reductionSum<<<blocksPerGrid, BLOCK_SIZE>>>(d_input, d_output, N);

    // 第二次规约 (如果 Block 数量大于 1)
    if (blocksPerGrid > 1) {
        reductionSum<<<1, BLOCK_SIZE>>>(d_output, d_output, blocksPerGrid);
    }

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum = %.2f (expected %d)\n", result, N);

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
```

---

### 2.3 CUDA 内存管理

#### 2.3.1 内存分配 API

```cuda
// Device 内存分配
cudaMalloc(void **devPtr, size_t size);
cudaFree(void *devPtr);

// Host 内存分配
cudaMallocHost(void **ptr, size_t size);  // Pinned Memory
cudaFreeHost(void *ptr);

// Unified Memory (CUDA 6.0+)
cudaMallocManaged(void **devPtr, size_t size);
```

#### 2.3.2 内存复制

```cuda
cudaMemcpy(void *dst, const void *src, size_t count,
           cudaMemcpyKind kind);

// kind 可以是:
// - cudaMemcpyHostToDevice
// - cudaMemcpyDeviceToHost
// - cudaMemcpyDeviceToDevice
// - cudaMemcpyHostToHost
```

#### 2.3.3 Pinned Memory vs Pageable Memory

**Pageable Memory** (默认):
```cuda
float *h_data = (float*)malloc(size);
// 可能被换出到磁盘，复制时需要先锁页
```

**Pinned Memory** (锁页内存):
```cuda
float *h_data;
cudaMallocHost(&h_data, size);
// 不会被换出，复制速度更快 (可达 PCIe 带宽上限)
```

**性能对比**:
- Pageable Memory: ~3-4 GB/s
- Pinned Memory: ~12-16 GB/s (PCIe 3.0 x16)

---

### 2.4 构建和编译

#### 2.4.1 使用 nvcc 编译

```bash
# 基本编译
nvcc hello.cu -o hello

# 指定架构 (本仓库支持的架构)
nvcc -arch=sm_80 hello.cu -o hello  # A100
nvcc -arch=sm_86 hello.cu -o hello  # RTX 3060
nvcc -arch=sm_89 hello.cu -o hello  # RTX 4060

# 优化编译
nvcc -O3 -arch=sm_80 hello.cu -o hello

# 包含头文件和库
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcublas hello.cu -o hello
```

#### 2.4.2 使用 CMake (推荐)

本仓库使用 CMake 构建系统，参考 `matmul/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.18)
project(cuda_project CUDA CXX)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 自动检测 GPU 架构
set(CMAKE_CUDA_ARCHITECTURES native)

# 查找 CUDA
find_package(CUDA REQUIRED)

# 添加可执行文件
add_executable(gemm_test main.cu)

# 链接 cuBLAS
target_link_libraries(gemm_test cublas)
```

**构建流程**:
```bash
cd matmul
mkdir -p build && cd build
cmake ..
make
./gemm_test
```

---

## 🔧 第三部分: 性能分析工具 (晚上)

### 3.1 性能分析工具概览

| 工具 | 功能 | 使用场景 |
|-----|------|---------|
| **nvprof** (已弃用) | 基础性能分析 | CUDA 10.x 及以前 |
| **nsys** (Nsight Systems) | 系统级性能分析 | CPU-GPU 交互、Kernel 启动 |
| **ncu** (Nsight Compute) | Kernel 级性能分析 | 单个 Kernel 深度优化 |
| **nvvp** (Visual Profiler) | 可视化分析 | CUDA 10.x 及以前 |

---

### 3.2 Nsight Systems (nsys)

**适用场景**: 系统级性能分析，查看 CPU 和 GPU 的时间线

#### 3.2.1 基本使用

```bash
# 运行并生成报告
nsys profile -o report ./gemm_test

# 生成的文件: report.nsys-rep (可用 Nsight Systems GUI 打开)
```

#### 3.2.2 常用选项

```bash
# 只追踪 CUDA API
nsys profile --trace=cuda ./program

# 追踪 CUDA + NVTX (代码标记)
nsys profile --trace=cuda,nvtx ./program

# 设置采样频率
nsys profile --sample=cpu ./program
```

#### 3.2.3 代码中添加 NVTX 标记

```cuda
#include <nvToolsExt.h>

void myFunction() {
    nvtxRangePush("myFunction");
    // ... your code ...
    nvtxRangePop();
}

// 在 Kernel 外标记
nvtxRangePush("Matrix Multiplication");
matmulKernel<<<grid, block>>>(d_A, d_B, d_C);
cudaDeviceSynchronize();
nvtxRangePop();
```

#### 3.2.4 查看报告

```bash
# 命令行查看
nsys stats report.nsys-rep

# GUI 查看 (推荐)
nsys-ui report.nsys-rep
```

**查看内容**:
- Kernel 执行时间
- Memory Transfer 时间
- CPU-GPU 同步点
- API 调用开销

---

### 3.3 Nsight Compute (ncu)

**适用场景**: 单个 Kernel 的深度优化

#### 3.3.1 基本使用

本仓库已经提供了 Nsight Compute 的详细指南:
- 参考文档: `matmul/nsight_compute_guide.md`
- 示例脚本: `matmul/profile_example.sh`

```bash
# 分析所有 Kernel
ncu -o report ./gemm_test

# 只分析特定 Kernel
ncu --kernel-name matmulKernel -o report ./gemm_test

# 查看详细指标
ncu --set full -o report ./gemm_test
```

#### 3.3.2 关键指标

**内存指标**:
- `dram_throughput`: Global Memory 吞吐量
- `l2_cache_hit_rate`: L2 Cache 命中率
- `shared_memory_throughput`: Shared Memory 吞吐量

**计算指标**:
- `sm_efficiency`: SM 利用率
- `achieved_occupancy`: 实际占用率
- `warp_execution_efficiency`: Warp 执行效率

**Roofline 分析**:
- 判断是 Memory-bound 还是 Compute-bound

---

### 3.4 nvcc 编译选项

#### 3.4.1 常用选项

```bash
# 生成 PTX 中间代码
nvcc -ptx hello.cu -o hello.ptx

# 生成 SASS 汇编代码
nvcc -cubin hello.cu -o hello.cubin
cuobjdump -sass hello.cubin

# 查看寄存器使用
nvcc -Xptxas -v hello.cu

# 限制寄存器数量
nvcc --maxrregcount=32 hello.cu

# 优化级别
nvcc -O0  # 无优化
nvcc -O2  # 默认
nvcc -O3  # 最高优化

# 生成调试信息
nvcc -g -G hello.cu  # -g: host 调试, -G: device 调试
```

#### 3.4.2 架构相关选项

```bash
# 针对本仓库支持的架构
nvcc -arch=sm_80 hello.cu   # A100
nvcc -arch=sm_86 hello.cu   # RTX 3060
nvcc -arch=sm_89 hello.cu   # RTX 4060

# 生成多架构的 fatbin
nvcc -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     hello.cu
```

---

## 📝 第四部分: 作业 - 矩阵乘法优化

### 4.1 作业要求

实现并优化矩阵乘法 (GEMM - General Matrix Multiply):
```
C = A × B
其中 A: M×K, B: K×N, C: M×N
```

**实现版本**:
1. Naive 版本 (Global Memory 直接访问)
2. Shared Memory 优化版本
3. (可选) Tiling 优化版本

### 4.2 Naive 实现

```cuda
// naive_gemm.cu
#include <stdio.h>
#include <cuda_runtime.h>

#define M 1024
#define N 1024
#define K 1024

// Naive Matrix Multiplication
__global__ void naiveGemm(const float *A, const float *B, float *C,
                          int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // 分配 Host 内存
    float *h_A = (float*)malloc(bytesA);
    float *h_B = (float*)malloc(bytesB);
    float *h_C = (float*)malloc(bytesC);

    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    // 分配 Device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);

    // 启动 Kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);

    // 测量时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naiveGemm<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算性能
    float gflops = (2.0f * M * N * K * 1e-9) / (milliseconds / 1000.0f);
    printf("Naive GEMM: %.2f ms, %.2f GFLOPS\n", milliseconds, gflops);

    // 验证结果
    cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost);
    printf("C[0][0] = %.2f (expected %.2f)\n", h_C[0], (float)K);

    // 释放内存
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

**性能分析**:
- 每个线程需要从 Global Memory 读取 K 次
- 内存访问效率低
- 预期性能: ~50-100 GFLOPS (A100 理论峰值 19.5 TFLOPS FP32)

---

### 4.3 Shared Memory 优化

```cuda
// tiled_gemm.cu
#define TILE_SIZE 32

__global__ void tiledGemm(const float *A, const float *B, float *C,
                          int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 分块计算
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 Tile 到 Shared Memory
        if (row < m && t * TILE_SIZE + threadIdx.x < k) {
            tileA[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && t * TILE_SIZE + threadIdx.y < k) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // 计算 Tile 内的乘法
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads();
    }

    // 写回结果
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}
```

**优化点**:
- 使用 Shared Memory 减少 Global Memory 访问
- 内存复用: 每个数据被同一个 Block 内的多个线程使用
- 预期性能: ~500-1000 GFLOPS

---

### 4.4 参考本仓库的实现

本仓库已经包含了更多优化版本:
- `matmul/smcache/`: Shared Memory 优化
- `matmul/2dthreadtile/`: 2D Thread Tiling
- 参考: https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE

---

## 📚 扩展阅读

### 必读文档
1. **NVIDIA CUDA C Programming Guide** (Chapters 1-3)
   - https://docs.nvidia.com/cuda/cuda-c-programming-guide/

2. **CUDA Best Practices Guide**
   - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

3. **Nsight Compute Guide** (本仓库)
   - `matmul/nsight_compute_guide.md`

### 推荐视频
- **CS 149: Parallel Computing** - Stanford
- **NVIDIA DLI: CUDA 编程基础**

---

## ✅ 学习检查清单

### 上午任务
- [ ] 理解 GPU vs CPU 的架构差异
- [ ] 掌握 SM、Warp、内存层次结构
- [ ] 了解 Tensor Core 原理
- [ ] 熟悉本仓库支持的 GPU 架构配置

### 下午任务
- [ ] 编写并运行 Hello World Kernel
- [ ] 实现向量加法
- [ ] 实现矩阵加法
- [ ] 实现 Shared Memory 矩阵转置
- [ ] 实现规约求和

### 晚上任务
- [ ] 学会使用 nvcc 编译选项
- [ ] 熟悉 nsys 和 ncu 的基本使用
- [ ] 实现 Naive GEMM
- [ ] 优化为 Tiled GEMM (Shared Memory)
- [ ] 使用 ncu 分析性能瓶颈

---

## 📊 学习输出

完成今天的学习后，你应该能够:

1. **解释 GPU 架构**
   - SM、Warp、内存层次
   - Tensor Core 工作原理

2. **编写基础 CUDA 程序**
   - Thread/Block/Grid 配置
   - 内存分配和复制
   - Kernel 函数编写

3. **使用 Shared Memory 优化**
   - 理解内存访问模式
   - 避免 Bank Conflict

4. **性能分析**
   - 使用 nsys 查看时间线
   - 使用 ncu 分析 Kernel
   - 理解 Memory-bound vs Compute-bound

---

## 下一步学习

**Day 2: CUDA 内存优化**
- Coalesced Access
- Bank Conflict 避免
- Pinned Memory / Unified Memory
- Tensor Core 编程 (WMMA API)

继续加油! 🚀
