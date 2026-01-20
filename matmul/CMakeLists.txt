cmake_minimum_required(VERSION 3.18)
project(GEMM_CUDA_Project LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 编译选项
# GCC/G++ 编译参数: -O3 优化级别，-std=c++11 标准
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

# NVCC 编译参数
# 自动检测 GPU 架构（推荐用于 CUDA 12.2+）
set(CMAKE_CUDA_ARCHITECTURES native)

# CUDA 选项说明:
# -g: 主机代码添加调试信息
# -G: 设备代码产生调试信息（会禁用大部分优化）
# -Xptxas -dlcm=ca: 启用 L1 缓存
# -Xptxas -dlcm=cg: 关闭 L1 缓存
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")

# 查找 CUDA 工具包
find_package(CUDAToolkit REQUIRED)

# 配置头文件搜索路径
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/naive)
include_directories(${PROJECT_SOURCE_DIR}/smcache)
include_directories(${PROJECT_SOURCE_DIR}/1dthreadtile)
include_directories(${PROJECT_SOURCE_DIR}/2dthreadtile)
include_directories(${PROJECT_SOURCE_DIR}/registercache)

# 收集所有 kernel 源文件
set(KERNEL_SOURCES
    ${PROJECT_SOURCE_DIR}/naive/kernel1.cu
    ${PROJECT_SOURCE_DIR}/smcache/kernel2.cu
    ${PROJECT_SOURCE_DIR}/1dthreadtile/kernel3.cu
    ${PROJECT_SOURCE_DIR}/2dthreadtile/kernel4.cu
    ${PROJECT_SOURCE_DIR}/registercache/kernel5.cu
)

# 主程序使用 main.hpp (包含 main 函数)
set(MAIN_SOURCE ${PROJECT_SOURCE_DIR}/main.hpp)

# 可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

# 生成主可执行文件
add_executable(gemm_test ${MAIN_SOURCE})

# 将 main.hpp 设置为 CUDA 源文件
set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CUDA)

# 设置 CUDA 可分离编译（如果需要）
set_target_properties(gemm_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# 链接 CUDA 运行时库和 cuBLAS
target_link_libraries(gemm_test
    CUDA::cudart
    CUDA::cublas
)

# 打印配置信息
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit Include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "Executable output: ${EXECUTABLE_OUTPUT_PATH}")
