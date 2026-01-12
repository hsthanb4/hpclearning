cmake_minimum_required(VERSION 3.10)
project(GEMM_CUDA_Project LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 自动检测 GPU 架构（针对 CUDA 12.2 非常推荐）
set(CMAKE_CUDA_ARCHITECTURES native)

# 查找必要的 CUDA 库（比如 cuBLAS）
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(gemm_test main.cu)

# 如果你的代码用了 cuBLAS 等库，需要链接它们
target_link_libraries(gemm_test PRIVATE CUDA::cublas)