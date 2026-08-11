// ============================================================================
// CUDA 算子综合练习册 · 总索引
// ----------------------------------------------------------------------------
// 12 个基础算子已按技能主题拆分到 4 个练习文件：
//
//   practice_01_elementwise.cu   Elementwise / 映射类（热身）
//      01  Vector Add       c[i] = a[i] + b[i]
//      11  SiLU             y = x / (1 + exp(-x))
//      技能点：一维映射、grid-stride loop、合并访存、launch 四件套
//
//   practice_02_reduce.cu   Reduce 类（warp/block reduce 核心技能）
//      02  Reduce Sum       两阶段求和
//      03  GEMV             y = A*x + bias
//      04  RMSNorm          只统计 sum(x^2)
//      05  LayerNorm        统计 sum(x) 与 sum(x^2)
//      12  ArgMax           两阶段 (val, idx) reduce
//      技能点：shuffle reduce、smem 跨 warp 归并、两阶段 kernel、val+idx 对
//
//   practice_03_softmax.cu  Softmax 专题（数值稳定性 + online 合并）
//      06  Softmax Naive    一线程一输出，每行 O(N^2) 访存
//      07  Online Softmax   一个 block 一行，2-pass online (m,d) 合并
//      技能点：减 max 防溢出、(m,d) 合并公式 —— FlashAttention rescale 雏形
//
//   practice_04_tiling.cu   Shared Memory Tiling 专题（转置 + GEMM）
//      08  Transpose Naive  跨步写瓶颈
//      09  Transpose Tiled  32x32 tile + 33 列 padding 防 bank conflict
//      10  GEMM Tiled       16x16 tile，双 __syncthreads
//      技能点：strided -> coalesced、数据复用、同步位置、bank conflict
//
// 推荐练习顺序（每个文件内按编号顺序）：
//   第 1 天  practice_01（映射热身）+ practice_02 的 02 reduce_sum
//   第 2 天  practice_02 的 03/04/05（reduce 复用变形）+ 12 argmax
//   第 3 天  practice_03（softmax 从 naive 到 online）
//   第 4 天  practice_04（transpose 从 naive 到 tiled，再上 gemm_tiled）
//
// 每题三个问答题，先自己写答案，再对照源文件（01~12 号 .cu）核对。
//
// 答案版（与练习文件一一对应，先填空后对照）：
//   practice_answer_01_elementwise.cu
//   practice_answer_02_reduce.cu
//   practice_answer_03_softmax.cu
//   practice_answer_04_tiling.cu
// ============================================================================
