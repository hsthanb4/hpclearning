const fs = require("fs");
const path = require("path");

let seed = 17000;
let id = 0;

function base(type, x, y, width, height, options = {}) {
  return {
    id: options.id || `mlir_${id++}`,
    type,
    x,
    y,
    width,
    height,
    angle: 0,
    strokeColor: options.strokeColor || "#1e1e1e",
    backgroundColor: options.backgroundColor || "transparent",
    fillStyle: "solid",
    strokeWidth: options.strokeWidth || 2,
    strokeStyle: options.strokeStyle || "solid",
    roughness: 1,
    opacity: options.opacity ?? 100,
    groupIds: [],
    roundness: type === "rectangle" ? { type: 3 } : null,
    seed: seed++,
    version: 1,
    isDeleted: false,
    boundElements: null,
    updated: 1,
    link: null,
    locked: false,
  };
}

function text(value, x, y, width, height, options = {}) {
  return {
    ...base("text", x, y, width, height, {
      strokeColor: options.color || "#374151",
    }),
    text: value,
    fontSize: options.fontSize || 18,
    fontFamily: 5,
    textAlign: options.align || "center",
    verticalAlign: options.verticalAlign || "middle",
    containerId: null,
    originalText: value,
    autoResize: false,
    lineHeight: 1.25,
  };
}

function rect(x, y, width, height, options = {}) {
  return base("rectangle", x, y, width, height, options);
}

function box(label, x, y, width, height, options = {}) {
  return [
    rect(x, y, width, height, {
      strokeColor: options.strokeColor || "#3b82f6",
      backgroundColor: options.backgroundColor || "#dbe4ff",
      opacity: options.opacity ?? 100,
    }),
    text(label, x + 12, y + 10, width - 24, height - 20, {
      color: options.textColor || "#1f2937",
      fontSize: options.fontSize || 18,
      align: options.align || "center",
    }),
  ];
}

function arrow(x1, y1, x2, y2, options = {}) {
  return {
    ...base("arrow", x1, y1, x2 - x1, y2 - y1, {
      strokeColor: options.strokeColor || "#495057",
      strokeStyle: options.strokeStyle || "solid",
      strokeWidth: options.strokeWidth || 2,
    }),
    endArrowhead: options.endArrowhead ?? "arrow",
    startArrowhead: options.startArrowhead || null,
    points: [
      [0, 0],
      [x2 - x1, y2 - y1],
    ],
    lastCommittedPoint: null,
    startBinding: null,
    endBinding: null,
  };
}

function title(main, sub) {
  return [
    text(main, 90, 24, 1220, 36, { color: "#1e40af", fontSize: 28 }),
    text(sub, 90, 68, 1220, 28, { color: "#374151", fontSize: 18 }),
  ];
}

function doc(elements) {
  return {
    type: "excalidraw",
    version: 2,
    source: "https://excalidraw.com",
    elements,
    appState: {
      gridSize: null,
      viewBackgroundColor: "#ffffff",
    },
    files: {},
  };
}

function write(name, elements) {
  const out = path.join(__dirname, "..", "excalidraw", name);
  fs.writeFileSync(out, `${JSON.stringify(doc(elements), null, 2)}\n`, "utf8");
}

function roadmap() {
  const e = [
    ...title(
      "MLIR 入门学习路线总览",
      "目标：读懂 IR -> 会跑 mlir-opt -> 写 pass/rewrite -> 自定义 dialect -> 理解 lowering 到 LLVM/GPU"
    ),
  ];
  e.push(rect(40, 120, 1320, 600, { strokeColor: "#748ffc", backgroundColor: "#edf2ff", opacity: 35 }));
  const stages = [
    ["0. 环境与工具", "构建 llvm-project\nmlir-opt / mlir-translate\nmlir-tblgen / FileCheck", "#a5d8ff"],
    ["1. IR 心智模型", "Operation / Value\nRegion / Block\nType / Attribute / Location", "#b2f2bb"],
    ["2. Dialect 与 ODS", "命名空间和语义边界\nTableGen 描述 op\ntraits / interfaces / verifier", "#ffd8a8"],
    ["3. Pass 与 Rewrite", "PassManager pipeline\nPatternRewriter\ncanonicalize / cse / FileCheck", "#d0bfff"],
    ["4. Lowering 主线", "ConversionTarget\nTypeConverter\nlinalg -> memref/vector/gpu/llvm", "#c3fae8"],
  ];
  stages.forEach(([head, body, color], i) => {
    const x = 70 + i * 252;
    e.push(...box(`${head}\n\n${body}`, x, 170, 220, 210, { backgroundColor: color, strokeColor: "#3b82f6", fontSize: 17 }));
    if (i < stages.length - 1) e.push(arrow(x + 220, 275, x + 252, 275));
  });
  const weeks = [
    ["第 1 周", "读 IR + 跑工具\n验收：能解释 def-use"],
    ["第 2 周", "跟 Toy Tutorial\n验收：能写一个 ODS op"],
    ["第 3 周", "写 pass/rewrite\n验收：有 FileCheck 测试"],
    ["第 4 周", "做小 lowering\n验收：source dialect 到 target dialect"],
  ];
  weeks.forEach(([head, body], i) => {
    const x = 100 + i * 315;
    e.push(...box(`${head}\n${body}`, x, 455, 265, 120, { backgroundColor: "#fff3bf", strokeColor: "#f59f00", fontSize: 17 }));
    if (i < weeks.length - 1) e.push(arrow(x + 265, 515, x + 315, 515));
  });
  e.push(...box("最终小项目：极简 tensor DSL dialect\nmydsl.add / relu / matmul -> linalg / arith\n讲得清：为什么保留高层语义，为什么逐步 lowering", 310, 620, 780, 78, {
    backgroundColor: "#eebefa",
    strokeColor: "#ae3ec9",
    fontSize: 18,
  }));
  return e;
}

function irStructure() {
  const e = [
    ...title(
      "MLIR IR 结构与 Dialect 核心概念",
      "读 IR 时先找嵌套结构，再追 SSA def-use，最后看每个 op 属于哪个 dialect"
    ),
  ];
  e.push(rect(45, 120, 600, 520, { strokeColor: "#748ffc", backgroundColor: "#edf2ff", opacity: 35 }));
  e.push(text("嵌套结构", 85, 140, 520, 28, { color: "#1e40af", fontSize: 22 }));
  e.push(...box("builtin.module\n顶层 Operation", 210, 190, 270, 72, { backgroundColor: "#a5d8ff" }));
  e.push(...box("Region\nop 内部的嵌套区域", 230, 300, 230, 70, { backgroundColor: "#b2f2bb" }));
  e.push(...box("Block\n参数 + operation 列表", 250, 405, 190, 70, { backgroundColor: "#ffd8a8" }));
  e.push(...box("Operation\noperands / results / attrs / regions", 190, 515, 310, 78, { backgroundColor: "#d0bfff" }));
  e.push(arrow(345, 262, 345, 300));
  e.push(arrow(345, 370, 345, 405));
  e.push(arrow(345, 475, 345, 515));

  e.push(rect(705, 120, 655, 520, { strokeColor: "#69db7c", backgroundColor: "#ebfbee", opacity: 35 }));
  e.push(text("语义与数据流", 745, 140, 575, 28, { color: "#1e40af", fontSize: 22 }));
  e.push(...box("Dialect\n一组 op/type/attribute 的命名空间\n例：func / arith / linalg / memref / gpu / llvm", 760, 185, 520, 95, { backgroundColor: "#c3fae8" }));
  e.push(...box("Value\nSSA 值：op result 或 block argument\n每个 use 都指向一个定义", 760, 325, 245, 110, { backgroundColor: "#fff3bf" }));
  e.push(...box("Type / Attribute / Location\n类型约束、编译期常量、源码位置\nverifier 依赖这些不变量", 1035, 325, 245, 110, { backgroundColor: "#ffd8a8" }));
  e.push(...box("Traits / Interfaces\n把 op 语义抽象出来\n让 generic pass 可以理解不同 dialect", 860, 500, 320, 95, { backgroundColor: "#ffc9c9" }));
  e.push(arrow(1020, 280, 910, 325));
  e.push(arrow(1020, 280, 1160, 325));
  e.push(arrow(885, 435, 955, 500));
  e.push(arrow(1160, 435, 1080, 500));
  return e;
}

function passRewrite() {
  const e = [
    ...title(
      "MLIR Pass 与 Pattern Rewrite 工作流",
      "核心反馈循环：每加一个 pass，都看 IR before/after，用 verifier 和 FileCheck 把语义钉住"
    ),
  ];
  e.push(...box("输入 IR\n多 dialect 混合", 70, 190, 185, 95, { backgroundColor: "#a5d8ff" }));
  e.push(...box("PassManager\n嵌套 pipeline\nbuiltin.module(func.func(...))", 315, 170, 250, 135, { backgroundColor: "#d0bfff" }));
  e.push(...box("Analysis\n支配关系 / use-def\nshape / alias / side effect", 625, 190, 230, 95, { backgroundColor: "#fff3bf" }));
  e.push(...box("RewritePattern\nmatchAndRewrite\n替换 op DAG", 910, 170, 230, 135, { backgroundColor: "#b2f2bb" }));
  e.push(...box("输出 IR\nverifier 通过\nFileCheck 固化", 1200, 190, 150, 95, { backgroundColor: "#c3fae8" }));
  e.push(arrow(255, 238, 315, 238));
  e.push(arrow(565, 238, 625, 238));
  e.push(arrow(855, 238, 910, 238));
  e.push(arrow(1140, 238, 1200, 238));

  e.push(rect(80, 380, 1240, 260, { strokeColor: "#f59f00", backgroundColor: "#fff9db", opacity: 45 }));
  e.push(text("调试与验收清单", 120, 405, 1160, 30, { color: "#1e40af", fontSize: 22 }));
  const checks = [
    ["看变化", "-mlir-print-ir-before-all\n-mlir-print-ir-after-change"],
    ["查性能", "-mlir-timing\n-mlir-pass-statistics"],
    ["查失败", "-mlir-print-ir-after-failure\ncrash reproducer"],
    ["写测试", "lit + FileCheck\n检查 op 是否被替换"],
  ];
  checks.forEach(([h, b], i) => {
    e.push(...box(`${h}\n${b}`, 135 + i * 295, 465, 245, 110, { backgroundColor: "#ffffff", strokeColor: "#f59f00", fontSize: 17 }));
  });
  e.push(text("常见坑：pattern 反复匹配自己导致不收敛；改 result type 后没有处理 uses；没有 verifier/FileCheck 就合入 pass。", 125, 665, 1150, 36, {
    color: "#c2410c",
    fontSize: 18,
  }));
  return e;
}

function lowering() {
  const e = [
    ...title(
      "MLIR Lowering 到 GPU 和 LLVM 的主路径",
      "Lowering 是基于合法性目标的逐步转换，不是一次性把文本改成 LLVM IR"
    ),
  ];
  const pipeline = [
    ["前端 Dialect", "torch / tf / stablehlo\n或自定义 DSL", "#a5d8ff"],
    ["结构化算子", "linalg / tensor / arith\n保留计算语义", "#b2f2bb"],
    ["内存化", "bufferization -> memref\ntensor value 到 buffer", "#ffd8a8"],
    ["循环与向量", "scf / affine / vector\ntiling / fusion / vectorize", "#d0bfff"],
    ["GPU 目标", "gpu / nvgpu / nvvm\nlaunch / thread / intrinsic", "#ffc9c9"],
    ["底层出口", "llvm dialect\nLLVM IR / PTX / object", "#c3fae8"],
  ];
  pipeline.forEach(([h, b, c], i) => {
    const x = 55 + i * 218;
    e.push(...box(`${h}\n\n${b}`, x, 185, 185, 150, { backgroundColor: c, strokeColor: "#3b82f6", fontSize: 16 }));
    if (i < pipeline.length - 1) e.push(arrow(x + 185, 260, x + 218, 260));
  });

  e.push(rect(90, 420, 530, 210, { strokeColor: "#748ffc", backgroundColor: "#edf2ff", opacity: 40 }));
  e.push(text("Dialect Conversion 三件套", 130, 445, 450, 28, { color: "#1e40af", fontSize: 22 }));
  e.push(...box("ConversionTarget\n定义 legal / illegal / dynamic legal", 130, 500, 410, 42, { backgroundColor: "#ffffff", strokeColor: "#748ffc", fontSize: 17 }));
  e.push(...box("ConversionPattern\n把非法 op 改写成目标合法 op", 130, 555, 410, 42, { backgroundColor: "#ffffff", strokeColor: "#748ffc", fontSize: 17 }));
  e.push(...box("TypeConverter\n处理 tensor/memref/index/llvm type 等类型变化", 130, 610, 410, 42, { backgroundColor: "#ffffff", strokeColor: "#748ffc", fontSize: 17 }));

  e.push(rect(760, 420, 520, 210, { strokeColor: "#69db7c", backgroundColor: "#ebfbee", opacity: 40 }));
  e.push(text("学习时要能回答", 800, 445, 440, 28, { color: "#1e40af", fontSize: 22 }));
  e.push(text("1. 当前 op 为什么还需要保留高层语义？\n2. 目标 dialect 的合法边界是什么？\n3. 类型变化后 uses 如何保持安全？\n4. pass 前后 IR 是否通过 verifier？", 810, 495, 420, 110, {
    color: "#374151",
    fontSize: 18,
    align: "left",
    verticalAlign: "top",
  }));
  e.push(arrow(565, 595, 760, 595, { strokeStyle: "dashed" }));
  e.push(text("lowering 调试重点：合法性失败通常比 C++ 崩溃更有信息量，先读 IR 和 conversion debug 输出。", 185, 685, 1030, 34, {
    color: "#c2410c",
    fontSize: 18,
  }));
  return e;
}

fs.mkdirSync(path.join(__dirname, "..", "excalidraw"), { recursive: true });
write("MLIR_学习路线_总览.excalidraw", roadmap());
write("MLIR_IR结构与Dialect核心概念.excalidraw", irStructure());
write("MLIR_Pass与PatternRewrite工作流.excalidraw", passRewrite());
write("MLIR_Lowering到GPU_LLVM路径.excalidraw", lowering());

console.log("Generated 4 MLIR Excalidraw files.");
