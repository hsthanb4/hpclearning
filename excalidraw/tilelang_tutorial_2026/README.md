# TileLang 入门教程与 Excalidraw 图解

更新时间：2026-05-25

本目录是一套 TileLang 入门到面试复盘教程，包含详细 Markdown 教程和 6 张标准 `.excalidraw` 图。

## 文件

1. `TUTORIAL.md`：完整文字教程。
2. `00_tilelang_learning_roadmap.excalidraw`：学习路线。
3. `01_tilelang_programming_model.excalidraw`：编程模型。
4. `02_tilelang_kernel_anatomy.excalidraw`：kernel 解剖。
5. `03_tilelang_gemm_tile_flow.excalidraw`：GEMM tile 数据流。
6. `04_tilelang_tuning_debug_workflow.excalidraw`：调优与调试。
7. `05_tilelang_interview_map.excalidraw`：面试速查。
8. `generate_tilelang_tutorial.py`：重新生成教程和图。

## 资料来源

- TileLang docs: https://tilelang.com/
- TileLang GitHub: https://github.com/tile-ai/tilelang
- TileLang releases: https://github.com/tile-ai/tilelang/releases
- TileLang paper: https://arxiv.org/abs/2504.17577
- TVM docs: https://tvm.apache.org/docs/

截至 2026-05-25，TileLang 文档站页面显示版本为 0.1.10，GitHub releases 页面最新稳定标签为 v0.1.9。

## 练习题

1. 把同一个 GEMM tile 分别用 CUDA thread/block、Triton program、TileLang `T.Kernel` 三种语言描述，比较抽象粒度。
2. 修改 `BM/BN/BK/num_threads/num_stages`，记录 shared memory、寄存器压力和 latency 的变化。
3. 给一个非整除矩阵 shape，补齐边界保护，并用小 shape 对齐 PyTorch baseline。
4. 读生成代码或 IR，标出 global -> shared -> fragment -> global 的每一次数据移动。

## 面试题

1. TileLang 为什么强调 tiled dataflow，而不是让用户直接写每个 thread 的计算？
2. `T.copy`、`T.gemm`、`T.Pipelined` 分别对应 GPU kernel 里的哪些性能关键路径？
3. TileLang 和 Triton 都能写 GEMM，二者在抽象层、编译基础设施和调优方式上有什么不同？
4. 如果 TileLang kernel 性能不如预期，你会先检查 tile shape、memory scope、pipeline、生成代码还是 profiler 证据？
