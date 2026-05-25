# Triton 入门教程与 Excalidraw 图解

更新时间：2026-05-25

本目录包含一套 Triton 入门到面试复盘教程，配套 6 张标准 `.excalidraw` 图。

## 文件

1. `TUTORIAL.md`：完整文字教程、练习题和面试题。
2. `00_triton_learning_roadmap.excalidraw`：学习路线。
3. `01_triton_programming_model.excalidraw`：编程模型。
4. `02_triton_vector_add_softmax.excalidraw`：Vector Add 与 Softmax。
5. `03_triton_matmul_tile_flow.excalidraw`：Matmul tile 数据流。
6. `04_triton_autotune_debug_workflow.excalidraw`：Autotune、debug、profile。
7. `05_triton_interview_map.excalidraw`：面试速查。
8. `generate_triton_tutorial.py`：重新生成教程和图。

## 资料来源

- Triton docs: https://triton-lang.org/main/index.html
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- Triton installation: https://triton-lang.org/main/getting-started/installation.html
- Triton language API: https://triton-lang.org/main/python-api/triton.language.html
- Triton releases: https://github.com/triton-lang/triton/releases
- OpenAI Triton intro: https://openai.com/index/triton/

截至 2026-05-25，GitHub releases 页面显示最新 release 为 v3.7.0。

## 练习题

1. 把 Vector Add 改成 AXPY：`z = a * x + y`，比较 `BLOCK_SIZE=256/512/1024/2048` 的带宽。
2. 写 row-wise softmax，支持非 2 的幂列数，对齐 PyTorch baseline，并解释 mask 的必要性。
3. 写一个最小 matmul，修改 `BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages` 并记录延迟。
4. 将同一个 GEMM tile 用 Triton 和 TileLang 各画一遍，说明 program/grid 与 T.Kernel/T.copy 的对应关系。

## 面试题

1. Triton program 和 CUDA thread/block 的抽象差异是什么？
2. `tl.constexpr`、`tl.arange`、`tl.load mask` 分别解决什么问题？
3. matmul 中 program ordering 和 `GROUP_M` 为什么影响 L2 cache locality？
4. Triton 与 TileLang、CUDA、torch.compile/Inductor 的关系分别是什么？
