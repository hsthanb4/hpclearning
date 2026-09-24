# -*- coding: utf-8 -*-
"""
MLIR Compiler Track (8 Lessons)
Clean titles and 2 dedicated illustrations for each lesson:
- Fig 1: 核心操作概念图 (Core Operational Concept Diagram)
- Fig 2: 关键流程图 (Key Execution Workflow Diagram)
"""

MLIR_LESSONS = {
    "lesson01_ir_ssa.qmd": {
        "title": "第 1 课：IR 层级、SSA 与 Region",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph Module["ModuleOp (顶层容器)"]
        subgraph Func["func.func @kernel(%arg0: f32) -> f32"]
            subgraph Region["Region 0"]
                subgraph Block0["Block ^bb0 (入口块)"]
                    Op1["%0 = arith.addf %arg0, %arg0 : f32"]
                    Op2["%1 = arith.mulf %0, %cst : f32"]
                    Term["func.return %1 : f32"]
                end
            end
        end
    end
    Op1 -->|SSA定值传值| Op2
    Op2 -->|SSA定值传值| Term
```
<p class="caption" align="center"><em>图 1-1：MLIR 嵌套包含层次 (Operation → Region → Block → Operation) 与 SSA 定值-引用链</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Source["MLIR 文本源码 (.mlir)"] --> Lexer["词法/语法解析 (AsmParser)"]
    Lexer --> OpBuild["Operation 实例构建 & 操作数类型推导"]
    OpBuild --> Verify["Dialect & Trait 合法性校验 (verify())"]
    Verify --> SSA["SSA 符号表解析与 Block 前向引用绑定"]
    SSA --> InMem["内存 IR 树形数据结构 (MLIRContext)"]
```
<p class="caption" align="center"><em>图 1-2：MLIR Operation 解析验证与内存模型装载全流程</em></p>
"""
    },
    "lesson02_dialects_types.qmd": {
        "title": "第 2 课：Dialect、Type、Attribute 与接口",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph Dialect["MLIRContext (方言生态)"]
        Builtin["builtin (module, func)"]
        Arith["arith (addf, muli, cmpf)"]
        Linalg["linalg (matmul, generic)"]
        MemRef["memref (alloc, load, store)"]
        LLVM["llvm (ptr, getelementptr)"]
    end
    Linalg -.->|抽象降级| MemRef
    MemRef -.->|指针展开| LLVM
    Arith -.->|指令映射| LLVM
```
<p class="caption" align="center"><em>图 2-1：MLIR 多方言协同拓扑与类型系统共存体系</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Op["Operation 实例"] --> Query{"查询接口 dyn_cast<OpInterface>"}
    Query -->|命中| CallMethod["分发到特定方言 Trait / C++ 虚表实现"]
    Query -->|未实现| Fallback["返回 nullptr / 执行通用保底路径"]
    CallMethod --> Result["返回副作用分析 / 形状推导 / 内存别名结果"]
```
<p class="caption" align="center"><em>图 2-2：MLIR OpInterface 动态查询与多态解耦分发执行流程</em></p>
"""
    },
    "lesson03_ods.qmd": {
        "title": "第 3 课：ODS/TableGen 与自定义 Operation",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph ODS["TableGen 定义 (.td 文件)"]
        Def["def My_MatmulOp : My_Op<'matmul', [Pure, SameOperandsAndResultType]>"]
        Args["arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs)"]
        Res["results = (outs AnyTensor:$output)"]
        Asm["assemblyFormat = '$lhs `,` $rhs attr-dict `:` type($output)'"]
    end
    ODS --> GenCode["mlir-tblgen 驱动代码生成器"]
```
<p class="caption" align="center"><em>图 3-1：ODS 声明式算子规范 (Operation Definition Specification) 元模型</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    TD["算子定义规范 (*.td)"] -->|mlir-tblgen -gen-op-decls| IncH["算子头文件声明 (*.h.inc)"]
    TD -->|mlir-tblgen -gen-op-defs| IncCpp["算子实现代码 (*.cpp.inc)"]
    IncH & IncCpp --> CxxCompile["C++ 编译期集成"]
    CxxCompile --> DialectReg["MLIRContext 注册并供编译器调用"]
```
<p class="caption" align="center"><em>图 3-2：基于 mlir-tblgen 的自定义算子全自动构建与接入生命周期</em></p>
"""
    },
    "lesson04_patterns.qmd": {
        "title": "第 4 课：Pass、PatternRewriter 与贪心重写",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph Driver["GreedyPatternRewriteDriver"]
        Queue["待遍历 Operation 工作队列 (Worklist)"]
        Match{"Pattern::matchAndRewrite"}
        Queue -->|Pop Op| Match
        Match -->|匹配成功| Rewrite["Rewriter: replaceOp / eraseOp"]
        Match -->|不匹配| Next["考察队列下一个 Operation"]
        Rewrite -->|触发新修改| Requeue["将当前节点的用户与邻接 Op 重新压入队列"]
    end
```
<p class="caption" align="center"><em>图 4-1：MLIR 贪心模式重写驱动器 (Greedy Pattern Driver) 核心状态机</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Init["初始化 Pass 并在 Op 上启动 Driver"] --> Loop["弹出工作队列顶部 Operation"]
    Loop --> TryPatterns["遍历已注册 RewritePattern 列表"]
    TryPatterns --> Matched{"match 成功?"}
    Matched -->|是| Apply["执行 rewrite() 替换节点并通知监听器"]
    Matched -->|否| More{"队列为空?"}
    Apply --> More
    More -->|否| Loop
    More -->|是| Converged["IR 达到不动点 (Fixed Point)，Pass 顺利收敛"]
```
<p class="caption" align="center"><em>图 4-2：MLIR 模式重写与死代码消除的不动点迭代流程</em></p>
"""
    },
    "lesson05_conversion.qmd": {
        "title": "第 5 课：Dialect Conversion 与类型转换",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph ConversionFramework["方言转换框架核心三要素"]
        Target["ConversionTarget (合法性定义: Legal / Dynamic / Illegal)"]
        TypeConv["TypeConverter (类型映射: 如 tensor<f32> ➔ memref<f32>)"]
        Patterns["ConversionPattern (重写规则: 转换 Op 结构及操作数)"]
    end
    Target & TypeConv & Patterns --> Engine["applyPartialConversion / applyFullConversion"]
```
<p class="caption" align="center"><em>图 5-1：Dialect Conversion 目标合法性判定与类型映射模型</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    InputIR["源方言 IR (含 Illegal Op)"] --> TypeMap["TypeConverter 转换函数签名与操作数"]
    TypeMap --> PatternExec["Pattern 逐个生成新合法方言算子"]
    PatternExec --> Materialize["生成 unrealized_conversion_cast 桥接类型差异"]
    Materialize --> LegalCheck{"所有 Op 与 Type 均合法?"}
    LegalCheck -->|是| CleanCast["自动折叠并消除合法 cast，输出目标 IR"]
    LegalCheck -->|否| Fail["类型不匹配，Conversion 报错回滚"]
```
<p class="caption" align="center"><em>图 5-2：MLIR 方言降低 (Dialect Lowering) 与类型物化完整执行流程</em></p>
"""
    },
    "lesson06_bufferization.qmd": {
        "title": "第 6 课：Tensor 到 MemRef Bufferization 与别名",
        "fig1": """
::: {.img-card}
![](assets/figs/mlir_bufferization_footprint.png){width="85%"}
<p class="caption">图 6-1：One-Shot Bufferize 原地复用别名对峰值显存占用的削减（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart TD
    Analyze["遍历 AST 构建读写与别名图 (Alias / OpOperand Analysis)"] --> Detect{"存在写后读 (RAW) 冲突且无法重用 buffer?"}
    Detect -->|冲突| Alloc["插入显式内存拷贝 / memref.alloc 分配新缓冲区"]
    Detect -->|无冲突| InPlace["安全复用输入缓存 (In-place Bufferization) 避免拷贝"]
    Alloc & InPlace --> RewriteOps["将 Tensor 算子替换为等价 MemRef 算子"]
    RewriteOps --> Dealloc["BufferDeallocation Pass 插入释放，防止内存泄漏"]
```
<p class="caption" align="center"><em>图 6-2：One-Shot Bufferize 原地复用别名判定与内存分配下发流程</em></p>
"""
    },
    "lesson07_llvm_lowering.qmd": {
        "title": "第 7 课：逐级 Lowering 到 LLVM Dialect",
        "fig1": """
::: {.img-card}
![](assets/figs/mlir_lowering_latency.png){width="85%"}
<p class="caption">图 7-1：MLIR 渐进式多级降低与代码生成阶段性能加速比（基于 scientific-figure-making 绘制）</p>
:::
""",
        "fig2": """
```{mermaid}
flowchart LR
    MLIR["MLIR (LLVM Dialect)"] --> MLIRTrans["mlir-translate --mlir-to-llvmir"]
    MLIRTrans --> LLVMIR["标准 LLVM IR 模块 (llvm::Module)"]
    LLVMIR --> Opt["LLVM 优化管线 (llc -O3 / 向量化 / 寄存器分配)"]
    Opt --> Target["编译为平台原生目标码: x86/ARM ELF 或 NVIDIA PTX/CUBIN"]
```
<p class="caption" align="center"><em>图 7-2：从 LLVM Dialect 到底层硬件二进制目标代码的翻译运行流程</em></p>
"""
    },
    "lesson08_transform_debug.qmd": {
        "title": "第 8 课：Transform Dialect、Pipeline 调试与面试设计",
        "fig1": """
```{mermaid}
flowchart TD
    subgraph TransformScript["Transform Dialect 脚本"]
        Match["%target = transform.structured.match ops{['linalg.matmul']}"]
        Tile["%tiled, %loops = transform.structured.tile_to_scf_for %target [32, 32]"]
        Vector["transform.structured.vectorize %tiled"]
    end
    subgraph PayloadIR["Payload 计算图"]
        Matmul["linalg.matmul"] --> TiledLoops["scf.for i ... scf.for j ... linalg.matmul (32x32)"]
        TiledLoops --> VectorOps["vector.contract / vector.fma"]
    end
    TransformScript -.->|解释器解释并就地变异| PayloadIR
```
<p class="caption" align="center"><em>图 8-1：Transform Dialect 元调度脚本与被变换 Payload IR 的解耦驱动体系</em></p>
""",
        "fig2": """
```{mermaid}
flowchart LR
    Bug["Pass 出现崩溃或语义错误"] --> Opt["启动调试: mlir-opt --pass-pipeline=..."]
    Opt --> Print["添加 --mlir-print-ir-after-all 输出各阶段 IR"]
    Print --> CrashRepro["生成 --mlir-generate-crash-reproducer 独立复现用例"]
    CrashRepro --> Delta["利用 mlir-reduce 自动化精简缩小问题 IR 块"]
    Delta --> Fix["定位违背 Invariant 的 Pass 并修复断言"]
```
<p class="caption" align="center"><em>图 8-2：MLIR 编译管线排错、最小化复现与诊断全流程</em></p>
"""
    }
}
