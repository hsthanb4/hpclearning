# MLIR 入门详细教程

这份教程面向已经熟悉一点 C++、编译器或 AI Infra 的学习者。目标不是把 MLIR 所有 API 背下来，而是建立一条能动手的路径：先读懂 IR，再会跑 `mlir-opt`，然后写 pass、写 rewrite、写一个小 dialect，最后理解 lowering 到 LLVM/GPU 的主线。

配套 Excalidraw：

- [MLIR 学习路线总览](../excalidraw/MLIR_学习路线_总览.excalidraw)
- [MLIR IR 结构与 Dialect 核心概念](../excalidraw/MLIR_IR结构与Dialect核心概念.excalidraw)
- [MLIR Pass 与 Pattern Rewrite 工作流](../excalidraw/MLIR_Pass与PatternRewrite工作流.excalidraw)
- [MLIR Lowering 到 GPU 和 LLVM 路径](../excalidraw/MLIR_Lowering到GPU_LLVM路径.excalidraw)

## 1. 先建立正确心智模型

MLIR 是一个多层级 IR 框架。它不像 LLVM IR 那样主要代表较低层的通用机器无关 IR，而是允许不同抽象层共存：高层 tensor/dataflow、结构化 loop、buffer/memref、vector、gpu/nvvm、llvm dialect 等可以在同一个系统里逐步转换。

入门时先记住这 5 个词：

- `Operation`：MLIR 的基本节点。函数、常量、加法、for loop、gpu launch 都是 op。
- `Value`：SSA 值。它要么来自某个 op 的 result，要么是 block argument。
- `Region`：op 里面可以包含 region，用来表达嵌套结构。
- `Block`：region 由 block 组成，block 里放一串 operation。
- `Dialect`：一组 operation/type/attribute 的命名空间和语义边界，例如 `func`、`arith`、`linalg`、`memref`、`vector`、`gpu`、`llvm`。

一句话理解：MLIR = 可扩展 Operation 图 + SSA 值流 + 嵌套 Region/Block + 多 Dialect + Pass/Rewrite/Conversion 框架。

## 2. 环境准备

官方构建入口是 `llvm-project`，开启 `LLVM_ENABLE_PROJECTS=mlir`。Linux/WSL 上建议先用 Release + assertions，不要一开始上 Debug 全量构建。

```bash
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --target check-mlir
```

Windows 也可以直接构建，但如果目的是学习 AI compiler/GPU lowering，推荐 WSL2 或 Linux 机器。你需要的第一批工具是：

- `mlir-opt`：跑 pass、看 IR 变化。
- `mlir-translate`：在 MLIR 与 LLVM IR 等格式之间转换。
- `mlir-tblgen`：从 TableGen/ODS 生成 dialect/op/pass 相关代码。
- `FileCheck`：写 compiler test 时检查输出模式。

本机当前没有检测到 `mlir-opt`，所以这份仓库里的教程是学习材料和图谱，命令需要在装好 LLVM/MLIR 后运行。

## 3. 第一个可读 IR

先看一个普通函数：

```mlir
module {
  func.func @add_one(%arg0: f32) -> f32 {
    %c1 = arith.constant 1.0 : f32
    %0 = arith.addf %arg0, %c1 : f32
    return %0 : f32
  }
}
```

读法：

- `module` 是顶层 op，里面有 region/block。
- `func.func` 是 func dialect 的函数 op。
- `%arg0` 是 block argument。
- `%c1` 和 `%0` 是 op result。
- `arith.constant`、`arith.addf` 属于 arith dialect。
- `: f32` 是类型约束。

建议第一天只做一件事：拿 10 个 `.mlir` 文件，逐行标出 op、operand、result、type、attribute、region、block。能读懂 IR 比先写 C++ API 更重要。

## 4. 用 `mlir-opt` 建立反馈循环

有工具后，把上面的 IR 保存成 `add_one.mlir`，先跑：

```bash
mlir-opt add_one.mlir
mlir-opt add_one.mlir -canonicalize
mlir-opt add_one.mlir -cse
mlir-opt add_one.mlir -verify-diagnostics
```

然后学习 pass pipeline：

```bash
mlir-opt add_one.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))'
```

调试 pass 时常用：

```bash
mlir-opt input.mlir -pass-pipeline='builtin.module(func.func(cse))' \
  -mlir-print-ir-before-all \
  -mlir-print-ir-after-all \
  -mlir-print-ir-after-change
```

这一步的关键不是记住每个 flag，而是养成习惯：每加一个 pass，都要看 pass 前后的 IR，确认语义和类型是否还成立。

## 5. Dialect 是 MLIR 的核心抽象

Dialect 解决的问题是：不同层级的语义不必强行塞进同一种 IR。你可以用高层 dialect 保留领域语义，再逐步 lowering 到更低层。

常见层级可以这样记：

- 前端语义：`torch`、`tf`、`mhlo/stablehlo`、自定义 DSL dialect。
- 张量计算：`linalg`、`tensor`、`arith`、`math`。
- 结构化控制流和内存：`scf`、`affine`、`memref`、`bufferization`。
- SIMD/GPU：`vector`、`gpu`、`nvgpu`、`nvvm`、`amdgpu`。
- 底层出口：`llvm` dialect，再到 LLVM IR/object/PTX 等。

学习 dialect 时重点看 4 件事：

- 这个 dialect 的 op 表达什么语义？
- op 的 operands/results/attributes/regions 是什么？
- verifier 保证什么不变量？
- 它通常 lowering 到哪些 dialect？

## 6. ODS 和 TableGen 入门

多数 op 不会纯手写 C++ 类，而是用 ODS/TableGen 描述，再由 `mlir-tblgen` 生成样板代码。一个 op 定义通常包含：

- op 名称和 dialect 命名空间。
- operands/results/attributes/regions/successors。
- traits 和 interfaces。
- assembly format。
- verifier/parser/printer/builder 的自定义逻辑。

非常简化的样子：

```tablegen
def Toy_AddOp : Toy_Op<"add"> {
  let summary = "toy add operation";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

不要一开始追求完整 out-of-tree 项目。先读 Toy Tutorial 的 Ch2/Ch3，再看现有 dialect 的 `.td` 文件。理解 ODS 之后，再补 C++ glue code。

## 7. Pass、Pattern Rewrite、Canonicalization

MLIR 的变换通常分三类：

- 普通 pass：遍历 IR，做分析或改写。
- pattern rewrite：把某类 op DAG 匹配后替换成另一组 op。
- dialect conversion：带合法性目标和类型转换的 lowering。

最小心智模型：

```text
input IR
  -> PassManager 调度
  -> Pass 中收集 RewritePattern
  -> PatternRewriter 匹配并替换 op
  -> verifier 检查 IR
  -> output IR
```

写 rewrite 时要避免两个常见错误：

- 只改了 op，没处理 uses/type，导致 verifier 失败。
- pattern 可以反复匹配自己，导致 greedy rewrite 不收敛。

建议练习顺序：

1. 先用现成 `-canonicalize` 和 `-cse` 观察 IR。
2. 写一个只删除冗余 op 的 canonicalization pattern。
3. 写一个把自定义 `toy.add` lowering 到 `arith.addf` 的 conversion pattern。
4. 再碰 type converter 和 full conversion。

## 8. Dialect Conversion 和 Lowering

Lowering 不是简单的字符串替换，而是“把非法 op 转成目标合法 op”。核心组件：

- `ConversionTarget`：定义哪些 dialect/op 合法、动态合法或非法。
- `RewritePattern` / `ConversionPattern`：把非法 op 改写成合法 op。
- `TypeConverter`：把源类型转换成目标类型，必要时插入 materialization。

一个典型 AI compiler lowering 路径：

```text
frontend graph dialect
  -> linalg/tensor/arith
  -> bufferization -> memref
  -> scf/affine/vector
  -> gpu/nvgpu/nvvm 或 llvm
  -> LLVM IR / PTX / object
```

学习重点不是“背固定 pipeline”，而是能解释每一步为什么存在：

- `linalg` 保留结构化算子语义，便于 tiling/fusion/vectorization。
- `bufferization` 解决 tensor value 到 memory buffer 的过渡。
- `vector` 表达 SIMD 或向量化意图。
- `gpu/nvgpu/nvvm` 表达 GPU launch、thread/block、target-specific intrinsic。
- `llvm` dialect 对应底层 LLVM IR 出口。

## 9. 4 周入门路线

### 第 1 周：读 IR 和跑工具

目标：能读懂普通 `.mlir` 文件，能用 `mlir-opt` 跑 pipeline。

任务：

- 构建 MLIR 或使用已有 LLVM/MLIR 工具链。
- 阅读 LangRef 的 high-level structure。
- 跑 `mlir-opt -canonicalize -cse`。
- 学会 `-mlir-print-ir-before-all`、`-mlir-print-ir-after-all`、`-mlir-timing`。

验收：

- 能说出 Operation/Value/Region/Block/Dialect 的关系。
- 能解释一个函数 IR 中每个 `%0` 是谁定义、谁使用。

### 第 2 周：跟 Toy Tutorial

目标：知道前端 AST 如何生成 MLIR，以及自定义 dialect 为什么需要 ODS。

任务：

- 跑 Toy Ch1-Ch3。
- 读 `examples/toy` 里的 `.td`、MLIRGen、pass。
- 对照生成前后的 IR。

验收：

- 能解释 Toy dialect 的 op 语义。
- 能写出一个简单 op 的 operands/results/assembly format。

### 第 3 周：Pass 和 Rewrite

目标：会写一个小 pass，会写一个小 rewrite。

任务：

- 阅读 PatternRewriter 文档。
- 写 canonicalization pattern。
- 使用 `mlir-opt` 打印 pass 前后 IR。
- 写 lit/FileCheck 测试。

验收：

- 能解释 greedy rewrite 为什么可能不收敛。
- 能用 `FileCheck` 检查某个 op 被替换掉。

### 第 4 周：Lowering 主线

目标：知道 dialect conversion 的合法性模型，能做一个小 lowering。

任务：

- 阅读 Dialect Conversion 文档。
- 写 `toy.add -> arith.addf` 或 `mydialect.matmul -> linalg.matmul`。
- 理解 `ConversionTarget`、`TypeConverter`、`applyPartialConversion` 和 `applyFullConversion`。

验收：

- 能画出 source dialect 到 target dialect 的合法性边界。
- 能解释为什么 type-changing rewrite 不能随便替换 use。

## 10. 面试和项目导向的练习题

如果目标是 AI Infra / compiler 面试，不要只停留在“读过文档”。建议做一个小项目：

项目题：实现一个极简 tensor DSL dialect，然后 lowering 到 `linalg`。

范围：

- 支持 `mydsl.add`、`mydsl.relu`、`mydsl.matmul` 三个 op。
- 用 ODS 定义 op。
- 写 verifier 检查输入输出 rank/type。
- 写 canonicalization：`relu(relu(x)) -> relu(x)`。
- 写 lowering：`mydsl.add -> linalg.generic`，`mydsl.matmul -> linalg.matmul`。
- 用 lit/FileCheck 写 3 个测试。

可讲述点：

- 为什么先保留 DSL 语义，再 lowering。
- traits/interfaces 帮助 generic pass 理解 op。
- conversion target 如何定义合法 IR。
- bufferization 前后 tensor/memref 的差异。
- pass pipeline 如何调试。

## 11. 常见坑

- 先写 C++ API，后读 IR：顺序反了。先会读 textual IR。
- 以为 dialect 越低越好：高层 dialect 的价值是保留优化所需的语义。
- 忽略 verifier：MLIR 很依赖 op verifier 和 traits/interfaces 保证不变量。
- 把 lowering 当成单个 pass：真实 pipeline 通常是多步 partial lowering。
- 不写 lit/FileCheck：compiler 代码没有输出测试，很容易“看起来能跑但语义漂移”。
- 不看 pass 前后 IR：调试 MLIR 最直接证据就是 IR diff。

## 12. 推荐阅读顺序

1. MLIR Getting Started：先把工具链搭起来。
2. MLIR Language Reference：重点看 Operation/Value/Region/Block/Dialect。
3. Understanding the IR Structure：用 C++ traversal 加深 IR 嵌套结构。
4. Using `mlir-opt`：形成工具反馈循环。
5. Toy Tutorial：理解从 AST 到 dialect，再到 lowering。
6. ODS / Operation Definition Specification：学习 TableGen 定义 op。
7. Pattern Rewriter：学习 canonicalization 和普通 rewrite。
8. Pass Infrastructure：学习 pass manager、pipeline、调试和统计。
9. Dialect Conversion：学习 lowering 的合法性模型。
10. Linalg / Bufferization / GPU / LLVM dialect：按 AI compiler 方向继续深入。

## 13. 官方资料

- MLIR Getting Started: https://mlir.llvm.org/getting_started/
- MLIR Language Reference: https://mlir.llvm.org/docs/LangRef/
- Understanding the IR Structure: https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/
- Using `mlir-opt`: https://mlir.llvm.org/docs/Tutorials/MlirOpt/
- Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/
- Creating a Dialect: https://mlir.llvm.org/docs/Tutorials/CreatingADialect/
- Operation Definition Specification: https://mlir.llvm.org/docs/DefiningDialects/Operations/
- Pattern Rewriter: https://mlir.llvm.org/docs/PatternRewriter/
- Pass Infrastructure: https://mlir.llvm.org/docs/PassManagement/
- Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/
- Bufferization: https://mlir.llvm.org/docs/Bufferization/
- Linalg Dialect: https://mlir.llvm.org/docs/Dialects/Linalg/
- GPU Dialect: https://mlir.llvm.org/docs/Dialects/GPU/
