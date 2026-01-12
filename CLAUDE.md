# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an HPC (High Performance Computing) learning repository focused on GPU computing with CUDA. The codebase contains implementations and experiments for optimized GPU kernels, primarily focusing on matrix multiplication (GEMM) and Flash Attention.

## Build System

The repository uses CMake for building CUDA projects:

```bash
# Navigate to project directory
cd matmul

# Configure and build
mkdir -p build && cd build
cmake ..
make

# Run the executable
./gemm_test
```

**Key CMake Configuration Details:**
- C++ and CUDA standard: C++11
- GPU architecture detection: Uses `CMAKE_CUDA_ARCHITECTURES native` to auto-detect the current GPU
- Dependencies: Requires CUDA Toolkit and cuBLAS
- The CMakeLists.txt expects a `main.cu` file as the executable source

## GPU Architecture Reference

When working with CUDA code, target architectures matter. The repository documents these mappings:

- A100: `sm_80` (Ampere)
- RTX 3060: `sm_86` (Ampere)
- Orin: `sm_87` (Ampere)
- RTX 4060: `sm_89` (Ada Lovelace)
- Thor: `sm_100` / `sm100a` (Blackwell)

## Code Organization

### matmul/
Contains GEMM (General Matrix Multiply) implementations and optimizations:
- **smcache/**: Shared memory caching implementations for matrix multiplication
- **2dthreadtile/**: 2D thread tiling strategies for GEMM optimization
- Reference implementation from: https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE

### fa/
Flash Attention related code and documentation:
- **fa.md**: Contains CUDA code snippets and patterns for global-to-shared memory transfers
- Includes examples using `cp.async.cg.shared.global` PTX instructions for async memory copies
- Works with `nv_bfloat16` data types

## CUDA Programming Patterns

### Global to Shared Memory Transfer

The codebase uses asynchronous copy patterns with PTX inline assembly for efficient data movement:

```c++
// Template pattern: <HEIGHT, WIDTH, TB_SIZE>
// Uses cp.async.cg.shared.global for 16-byte aligned transfers
// Stride and thread indexing handled per iteration
```

This pattern is critical for optimizing memory bandwidth in tiled matrix operations.

## Development Notes

- The repository is in Chinese, with documentation and comments primarily in Chinese
- Build artifacts are in `build/` directories (should be ignored)
- The codebase is a learning resource, focused on progressive optimization techniques for GPU computing
