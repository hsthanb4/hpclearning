from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent


PALETTE = {
    "title": "#1e40af",
    "text": "#243042",
    "muted": "#64748b",
    "line": "#2563eb",
    "blue": "#bfdbfe",
    "green": "#bbf7d0",
    "orange": "#fed7aa",
    "purple": "#ddd6fe",
    "red": "#fecaca",
    "yellow": "#fef3c7",
    "cyan": "#99f6e4",
    "pink": "#fbcfe8",
    "slate": "#e2e8f0",
}


class Diagram:
    def __init__(self, title: str, subtitle: str, width: int = 1700, height: int = 1160) -> None:
        self.title = title
        self.subtitle = subtitle
        self.width = width
        self.height = height
        self.elements: list[dict] = []
        self._counter = 0
        self.title_block()

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def _base(self, element_type: str, x: float, y: float, w: float, h: float, **style: object) -> dict:
        seed = 39000 + self._counter * 149
        base = {
            "id": self._id(element_type),
            "type": element_type,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": style.pop("strokeColor", "#1f2937"),
            "backgroundColor": style.pop("backgroundColor", "transparent"),
            "fillStyle": style.pop("fillStyle", "solid"),
            "strokeWidth": style.pop("strokeWidth", 2),
            "strokeStyle": style.pop("strokeStyle", "solid"),
            "roughness": style.pop("roughness", 1),
            "opacity": style.pop("opacity", 100),
            "groupIds": [],
            "roundness": {"type": 3} if element_type in {"rectangle", "diamond"} else None,
            "seed": seed,
            "version": 1,
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
        }
        if base["roundness"] is None:
            base.pop("roundness")
        base.update(style)
        return base

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "transparent",
        stroke: str = "#1f2937",
        opacity: int = 100,
        stroke_style: str = "solid",
    ) -> None:
        self.elements.append(
            self._base(
                "rectangle",
                x,
                y,
                w,
                h,
                strokeColor=stroke,
                backgroundColor=fill,
                opacity=opacity,
                strokeStyle=stroke_style,
            )
        )

    def text(
        self,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        size: int = 18,
        color: str = PALETTE["text"],
        align: str = "center",
        valign: str = "middle",
    ) -> None:
        el = self._base(
            "text",
            x,
            y,
            w,
            h,
            strokeColor=color,
            backgroundColor="transparent",
            strokeWidth=1,
        )
        el.update(
            {
                "text": text,
                "fontSize": size,
                "fontFamily": 5,
                "textAlign": align,
                "verticalAlign": valign,
                "containerId": None,
                "originalText": text,
                "autoResize": False,
                "lineHeight": 1.25,
            }
        )
        self.elements.append(el)

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = PALETTE["line"],
        stroke_style: str = "solid",
        end: str | None = "arrow",
    ) -> None:
        dx = x2 - x1
        dy = y2 - y1
        el = self._base(
            "arrow",
            x1,
            y1,
            abs(dx),
            abs(dy),
            strokeColor=color,
            backgroundColor="transparent",
            strokeWidth=2,
            strokeStyle=stroke_style,
        )
        el.update(
            {
                "points": [[0, 0], [dx, dy]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": end,
            }
        )
        self.elements.append(el)

    def box(
        self,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "#1f2937",
        size: int = 18,
        color: str = PALETTE["text"],
    ) -> None:
        self.rect(x, y, w, h, fill, stroke)
        self.text(text, x + 14, y + 10, w - 28, h - 20, size=size, color=color)

    def label(self, text: str, x: float, y: float, w: float = 420, h: float = 40) -> None:
        self.text(text, x, y, w, h, size=22, color=PALETTE["title"], align="left")

    def title_block(self) -> None:
        self.text(self.title, 80, 34, self.width - 160, 42, size=30, color=PALETTE["title"])
        self.text(self.subtitle, 90, 83, self.width - 180, 34, size=17, color=PALETTE["muted"])
        self.arrow(90, 130, self.width - 90, 130, "#94a3b8", "dashed", end=None)

    def save(self, filename: str) -> None:
        data = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
            "files": {},
        }
        (OUT_DIR / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def roadmap() -> Diagram:
    d = Diagram("Triton 入门学习路线", "从第一个 @triton.jit 到能解释 matmul、softmax、attention kernel")
    phases = [
        ("0  背景", "复习 CUDA block warp\n理解 GPU memory hierarchy", PALETTE["blue"]),
        ("1  安装", "pip install triton\n确认 PyTorch CUDA 环境", PALETTE["cyan"]),
        ("2  Vector Add", "program_id arange mask\n掌握 launch grid", PALETTE["green"]),
        ("3  Fused Softmax", "一行一个 program\n用 tl.max tl.sum 减少访存", PALETTE["yellow"]),
        ("4  Matmul", "BLOCK_M/N/K tl.dot\nprogram 分组和 L2 复用", PALETTE["orange"]),
        ("5  Attention", "QK softmax PV 分块\n理解 FlashAttention 风格", PALETTE["purple"]),
    ]
    for i, (title, body, fill) in enumerate(phases):
        px = 90 + (i % 3) * 520
        py = 210 + (i // 3) * 295
        d.box(title, px, py, 390, 70, fill, "#2563eb", size=22)
        d.box(body, px, py + 90, 390, 125, "#ffffff", "#94a3b8", size=18)
        if i not in {2, 5}:
            d.arrow(px + 390, py + 35, px + 520, py + 35)
        if i == 2:
            d.arrow(px + 190, py + 215, px + 190, py + 295)
    d.label("学习验收标准", 95, 835)
    d.box("会从 PyTorch wrapper 调用 kernel\n能解释 grid lambda 和 meta-parameters", 110, 895, 440, 120, PALETTE["slate"], "#475569")
    d.box("会画 program 对应的数据 block\n知道 mask 防越界和 constexpr 的作用", 630, 895, 440, 120, PALETTE["slate"], "#475569")
    d.box("会用 triton.testing.do_bench 或 profiler\n比较 baseline、Triton、torch 结果", 1150, 895, 440, 120, PALETTE["slate"], "#475569")
    return d


def programming_model() -> Diagram:
    d = Diagram("Triton 编程模型", "Triton program 是比 CUDA thread 更粗的 block-level 单元")
    d.label("Host 侧", 90, 175)
    d.box("Python wrapper\n准备 torch tensors\n计算 grid", 95, 235, 350, 110, PALETTE["blue"], "#2563eb")
    d.box("kernel[grid](args,\nBLOCK_SIZE=..., num_warps=...)", 95, 390, 350, 120, PALETTE["blue"], "#2563eb")
    d.label("Kernel 侧", 590, 175)
    d.box("@triton.jit\n函数被 JIT 编译为 GPU kernel", 595, 235, 390, 110, PALETTE["green"], "#16a34a")
    d.box("tl.program_id(axis)\n当前 program 在 grid 中的位置", 595, 390, 390, 120, PALETTE["green"], "#16a34a")
    d.box("tl.arange / pointer arithmetic\n构造 block 内 offsets", 595, 545, 390, 120, PALETTE["green"], "#16a34a")
    d.label("数据与计算", 1130, 175)
    d.box("tl.load / tl.store\nmask + other 处理边界", 1135, 235, 390, 110, PALETTE["purple"], "#7c3aed")
    d.box("tl.dot / reductions / math ops\n在 block tensor 上表达计算", 1135, 390, 390, 120, PALETTE["purple"], "#7c3aed")
    d.box("compiler lowering\nTTIR -> TTGIR -> LLVM/PTX/AMDGPU", 1135, 545, 390, 120, PALETTE["purple"], "#7c3aed")
    for y in [290, 450]:
        d.arrow(445, y, 595, y)
        d.arrow(985, y, 1135, y)
    d.label("和 CUDA 的映射", 90, 760)
    d.box("CUDA thread\n细粒度线程\n显式 threadIdx/blockIdx", 115, 825, 330, 120, PALETTE["yellow"], "#d97706")
    d.box("Triton program\n一个 block-level 程序\n一次处理向量或矩阵 tile", 515, 825, 390, 120, PALETTE["orange"], "#ea580c")
    d.box("meta-parameters\nBLOCK_M/N/K num_warps num_stages\n决定编译特化和性能", 975, 825, 460, 120, PALETTE["red"], "#dc2626")
    d.arrow(445, 885, 515, 885)
    d.arrow(905, 885, 975, 885)
    return d


def vector_add_softmax() -> Diagram:
    d = Diagram("Vector Add 与 Fused Softmax", "先用两个基础 kernel 建立 block tensor 和 mask 心智模型")
    d.label("Vector Add", 90, 170)
    d.box("grid = ceildiv(N, BLOCK_SIZE)\n每个 program 处理一段连续元素", 105, 235, 430, 115, PALETTE["blue"], "#2563eb")
    d.box("offsets = pid * BLOCK_SIZE + arange\nmask = offsets < N", 105, 390, 430, 115, PALETTE["cyan"], "#0891b2")
    d.box("tl.load x y\nz = x + y\ntl.store mask", 105, 545, 430, 115, PALETTE["green"], "#16a34a")
    d.arrow(320, 350, 320, 390)
    d.arrow(320, 505, 320, 545)
    d.label("Fused Softmax", 765, 170)
    d.box("一个 program 处理一行\nBLOCK_SIZE 取 next_power_of_2", 775, 235, 430, 115, PALETTE["yellow"], "#d97706")
    d.box("load row\nx - max(x)\nexp / sum / normalize", 775, 390, 430, 115, PALETTE["orange"], "#ea580c")
    d.box("只读写一次 global memory\n比 PyTorch 多 op 路径更省 HBM", 775, 545, 430, 115, PALETTE["purple"], "#7c3aed")
    d.arrow(990, 350, 990, 390)
    d.arrow(990, 505, 990, 545)
    d.label("共同检查点", 90, 760)
    d.box("program_id 是否映射到正确数据块", 110, 825, 340, 105, PALETTE["slate"], "#475569")
    d.box("mask 是否覆盖尾部和 padding", 500, 825, 340, 105, PALETTE["slate"], "#475569")
    d.box("BLOCK_SIZE 是否影响 occupancy 和寄存器", 890, 825, 380, 105, PALETTE["slate"], "#475569")
    d.box("结果要和 torch baseline 对齐", 1320, 825, 270, 105, PALETTE["slate"], "#475569")
    return d


def matmul_flow() -> Diagram:
    d = Diagram("Triton Matmul Tile 数据流", "理解 BLOCK_M/N/K、tl.dot、program grouping 和 L2 复用")
    d.label("输出 tile", 90, 170)
    d.box("program_id 映射到 C 的\nBLOCK_M x BLOCK_N tile", 100, 235, 360, 120, PALETTE["green"], "#16a34a")
    d.box("pid_m, pid_n\n决定 A/B/C 的指针偏移", 100, 390, 360, 110, PALETTE["green"], "#16a34a")
    d.label("K 维循环", 585, 170)
    d.box("A block\nBLOCK_M x BLOCK_K", 600, 240, 250, 95, PALETTE["blue"], "#2563eb")
    d.box("B block\nBLOCK_K x BLOCK_N", 920, 240, 250, 95, PALETTE["blue"], "#2563eb")
    d.box("accumulator\nBLOCK_M x BLOCK_N\nfloat32", 730, 445, 310, 115, PALETTE["yellow"], "#d97706")
    d.arrow(725, 335, 800, 445)
    d.arrow(1045, 335, 970, 445)
    d.label("性能控制", 1250, 170)
    d.box("num_warps\n影响并行度和调度", 1260, 235, 310, 95, PALETTE["orange"], "#ea580c")
    d.box("num_stages\n影响软件流水和寄存器压力", 1260, 370, 310, 105, PALETTE["orange"], "#ea580c")
    d.box("GROUP_M\n让相邻 program 复用 B tile\n改善 L2 locality", 1260, 520, 310, 120, PALETTE["orange"], "#ea580c")
    d.label("常见面试追问", 90, 760)
    d.box("为什么 accumulator 用 fp32\n减少 K 维累加误差", 110, 825, 340, 110, PALETTE["purple"], "#7c3aed")
    d.box("为什么要分组 program order\n提高 L2 cache 复用", 500, 825, 340, 110, PALETTE["purple"], "#7c3aed")
    d.box("BLOCK_K 过大或过小分别怎样\n寄存器压力 vs 复用不足", 890, 825, 380, 110, PALETTE["purple"], "#7c3aed")
    d.box("mask 在非整除 M/N/K 时如何保护\nload/store 都要考虑", 1320, 825, 300, 110, PALETTE["purple"], "#7c3aed")
    return d


def autotune_debug() -> Diagram:
    d = Diagram("Triton Autotune 与 Debug 工作流", "正确性、benchmark、autotune、profile 要按证据闭环")
    steps = [
        ("写 baseline", "torch reference\n小 shape assert_close", PALETTE["blue"]),
        ("写 Triton", "先固定 BLOCK_SIZE\n只求索引正确", PALETTE["cyan"]),
        ("benchmark", "triton.testing.do_bench\nwarmup + repeat", PALETTE["green"]),
        ("autotune", "triton.Config\nkey 决定何时重新搜索", PALETTE["yellow"]),
        ("debug", "static_print device_print\nTriton interpreter", PALETTE["orange"]),
        ("profile", "Nsight/Proton\n看内存、寄存器、occupancy", PALETTE["purple"]),
    ]
    for i, (title, body, fill) in enumerate(steps):
        px = 100 + (i % 3) * 520
        py = 215 + (i // 3) * 300
        d.box(title, px, py, 370, 70, fill, "#2563eb", size=22)
        d.box(body, px, py + 88, 370, 120, "#ffffff", "#94a3b8", size=18)
        if i not in {2, 5}:
            d.arrow(px + 370, py + 35, px + 520, py + 35)
        if i == 2:
            d.arrow(px + 185, py + 208, px + 185, py + 300)
    d.label("不要犯的错", 90, 865)
    d.box("只测整除 shape\n尾部 mask 没覆盖", 110, 925, 350, 105, PALETTE["red"], "#dc2626")
    d.box("autotune kernel 有副作用\n没有 reset_to_zero 或 restore_value", 505, 925, 410, 105, PALETTE["red"], "#dc2626")
    d.box("只看平均耗时\n忽略 p95、冷启动、cache 状态", 960, 925, 390, 105, PALETTE["red"], "#dc2626")
    return d


def interview_map() -> Diagram:
    d = Diagram("Triton 面试速查图", "把 Triton 学习收束成可讲述的题目框架")
    d.label("一句话定义", 90, 170)
    d.box("Triton 是 Python-based GPU kernel DSL 和 compiler\n让用户用 block-level program 写 DNN 自定义算子\n常用于 PyTorch 生态中的 fusion 和 matmul/attention 优化", 105, 230, 720, 135, PALETTE["blue"], "#2563eb", size=20)
    d.label("必须讲清的机制", 900, 170)
    d.box("program_id 和 grid\n决定数据块映射", 900, 230, 300, 110, PALETTE["green"], "#16a34a")
    d.box("mask 和 pointer arithmetic\n决定边界安全", 1235, 230, 330, 110, PALETTE["green"], "#16a34a")
    d.box("meta-parameters\n决定编译特化和搜索空间", 900, 390, 300, 110, PALETTE["green"], "#16a34a")
    d.box("tl.dot / reductions\n决定 block tensor 计算", 1235, 390, 330, 110, PALETTE["green"], "#16a34a")
    d.label("常见比较题", 90, 560)
    d.box("vs CUDA\nTriton 更少样板 更适合块级 DNN kernel\nCUDA 控制更细", 110, 625, 430, 120, PALETTE["yellow"], "#d97706")
    d.box("vs TileLang\nTriton 生态更成熟\nTileLang 更强调 tiled dataflow + TVM 衔接", 620, 625, 460, 120, PALETTE["yellow"], "#d97706")
    d.box("vs torch.compile\nInductor 可生成 Triton\n手写 Triton 用于更可控的热点 kernel", 1160, 625, 430, 120, PALETTE["yellow"], "#d97706")
    d.label("回答模板", 90, 860)
    d.box("先画数据块\n再讲 program/grid\n再讲 tl.load/tl.store/tl.dot\n最后讲调优证据和边界条件", 110, 920, 1420, 95, PALETTE["slate"], "#475569", size=22)
    return d


TUTORIAL = """# Triton 入门详细教程

更新时间：2026-05-25

这份教程面向已经学过 CUDA、PyTorch、FlashAttention 或 TileLang 的读者。目标不是背 API，而是建立一个能写、能调、能在面试里讲清楚的 Triton kernel 心智模型。

资料核验：

- Triton docs: https://triton-lang.org/main/index.html
- Triton tutorials: https://triton-lang.org/main/getting-started/tutorials/
- Triton installation: https://triton-lang.org/main/getting-started/installation.html
- Triton language API: https://triton-lang.org/main/python-api/triton.language.html
- Triton releases: https://github.com/triton-lang/triton/releases
- OpenAI Triton intro: https://openai.com/index/triton/

截至 2026-05-25，Triton docs main 页面说明 Triton 是面向现代 GPU 的 Python-based parallel programming language and compiler；GitHub releases 页面显示最新 release 为 v3.7.0。

## 1. Triton 是什么

Triton 是一个用于写 GPU kernel 的 Python DSL 和编译器。它的核心抽象不是 CUDA 里的单个 thread，而是一个 block-level 的 program：一个 program 处理一块向量、矩阵 tile 或 attention tile。

一句话记忆：

> Triton 让你用 Python 写出接近 CUDA 性能的 DNN 自定义 kernel，同时把 thread 级细节上升到 block tensor 和 program grid。

适用场景：

- PyTorch 中热点 op 的 fusion。
- elementwise、reduction、softmax、layernorm。
- matmul、group GEMM、persistent matmul。
- attention、MoE dispatch、quantization/dequantization kernel。

## 2. 安装和学习顺序

官方文档给出的稳定安装方式是：

```bash
pip install triton
```

源码开发方式：

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
pip install -r python/requirements.txt
pip install -e .
make dev-install
make test-nogpu
```

如果要真正跑 GPU kernel，还需要有匹配的 PyTorch、GPU driver 和 CUDA/HIP 环境。本仓库当前提供学习材料和图解，不声明本机已经跑通 Triton。

推荐学习顺序：

1. Vector Add：理解 `@triton.jit`、`tl.program_id`、`tl.arange`、mask。
2. Fused Softmax：理解一行一个 program、reduction、global memory 读写减少。
3. Matmul：理解 `BLOCK_M/N/K`、`tl.dot`、program grouping、autotune。
4. LayerNorm / Attention：理解真实 DNN kernel 中的访存、数值稳定和融合。

对应图：`00_triton_learning_roadmap.excalidraw`

## 3. Triton 编程模型

一个 Triton kernel 通常有两部分：

- Host wrapper：准备 PyTorch tensor、计算 grid、传入 meta-parameters。
- JIT kernel：用 `@triton.jit` 写 block-level 程序。

典型结构：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(z_ptr + offsets, x + y, mask=mask)


def add(x, y):
    z = torch.empty_like(x)
    n = z.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, z, n, BLOCK_SIZE=1024)
    return z
```

关键点：

- `tl.program_id(0)` 表示当前 program 在 grid 第 0 维的位置。
- `tl.arange` 创建 block 内 offset 向量。
- `mask` 处理尾部非整除。
- `BLOCK_SIZE: tl.constexpr` 是编译期 meta-parameter，会触发特化。

对应图：`01_triton_programming_model.excalidraw`

## 4. Vector Add 与 Softmax

Vector Add 是最小入口，Fused Softmax 是第一个能体现性能价值的例子。

Vector Add 学三件事：

- 一个 program 处理连续的一段元素。
- pointer arithmetic 和 mask 必须正确。
- wrapper 的 grid lambda 决定 program 数量。

Fused Softmax 学三件事：

- 一行一个 program 能把多次 PyTorch op 读写融合成一次读和一次写。
- 数值稳定要先减 `max`。
- `BLOCK_SIZE` 通常取 next power of 2，但过大会带来寄存器压力。

对应图：`02_triton_vector_add_softmax.excalidraw`

## 5. Matmul 怎么理解

Matmul 是 Triton 面试和实战的核心。你要能画出：

- 一个 program 对应 C 的一个 `BLOCK_M x BLOCK_N` tile。
- K 维按 `BLOCK_K` 分块循环。
- A block 和 B block 通过 `tl.load` 进入 block tensor。
- `tl.dot` 累加到 fp32 accumulator。
- `tl.store` 写回 C。

常见 meta-parameters：

- `BLOCK_M` / `BLOCK_N`：输出 tile 大小。
- `BLOCK_K`：K 维分块大小。
- `num_warps`：每个 program 的 warp 数。
- `num_stages`：流水深度。
- `GROUP_M`：program ordering，影响 L2 cache 复用。

对应图：`03_triton_matmul_tile_flow.excalidraw`

## 6. Autotune、Debug 和 Profile

Triton 的调优要有证据闭环：

1. 先写 PyTorch reference。
2. 小 shape 用 `torch.testing.assert_close`。
3. 用 `triton.testing.do_bench` 做基础 benchmark。
4. 用 `@triton.autotune` 枚举 `triton.Config`。
5. 用 profiler 看 occupancy、register、memory throughput。

注意：`triton.autotune` 会多次运行 kernel。如果 kernel 修改输出，必须考虑 `reset_to_zero`、`restore_value` 或保证输出被每次覆盖，否则 benchmark 会污染结果。

对应图：`04_triton_autotune_debug_workflow.excalidraw`

## 7. 练习题

练习 1：Vector Add 变体。

- 支持 `z = a * x + y`。
- 尝试 `BLOCK_SIZE=256/512/1024/2048`。
- 记录 latency 和带宽。

练习 2：Row-wise Softmax。

- 支持非 2 的幂列数。
- 对齐 PyTorch 结果。
- 比较 PyTorch 多 op 和 fused Triton 的显存读写次数。

练习 3：Matmul。

- 实现最小 matmul。
- 加上 fp32 accumulator。
- 尝试不同 `BLOCK_M/N/K` 和 `num_warps`。

练习 4：LayerNorm。

- 一行一个 program。
- 先算 mean，再算 variance。
- 思考为什么输入列数太大时一个 program 不够。

练习 5：和 TileLang 对比。

- 选择同一个 GEMM tile。
- 用 Triton 解释 program/grid。
- 用 TileLang 解释 T.Kernel/T.copy/T.gemm。

## 8. 面试题

Q：Triton program 和 CUDA thread 的关系是什么？

A：Triton program 更像一个 block-level work unit，通常处理一个向量 block 或矩阵 tile。你不用显式写每个 thread 的 threadIdx，而是用 block tensor 和 compiler 映射到底层线程。

Q：为什么 Triton 里很多参数是 `tl.constexpr`？

A：这些参数参与编译期特化，比如 BLOCK_SIZE、BLOCK_M/N/K。编译器需要知道它们才能生成静态 shape 的 block tensor、展开循环和做优化。

Q：mask 为什么重要？

A：Triton program 通常按固定 block size 处理数据，真实 shape 不一定整除。mask 保证尾部 load/store 不越界，也保证 padding 不污染计算。

Q：Triton matmul 为什么要关注 program ordering？

A：不同 program 顺序会影响 A/B tile 在 L2 cache 中的复用。GROUP_M 这类策略让相邻 program 共享更多输入 tile，减少 HBM 访问。

Q：Triton 和 CUDA 怎么取舍？

A：Triton 更适合快速写 DNN block-level kernel 和 PyTorch 集成；CUDA 控制更细，适合极限手工优化、复杂同步或 Triton 表达困难的场景。

Q：Triton 和 TileLang 怎么比较？

A：Triton 生态成熟，PyTorch/Inductor 结合紧；TileLang 更强调 tiled dataflow、TVM IR 和调度体系衔接。面试里不要说谁替代谁，要说它们抽象边界不同。

## 9. 文件清单

- `00_triton_learning_roadmap.excalidraw`
- `01_triton_programming_model.excalidraw`
- `02_triton_vector_add_softmax.excalidraw`
- `03_triton_matmul_tile_flow.excalidraw`
- `04_triton_autotune_debug_workflow.excalidraw`
- `05_triton_interview_map.excalidraw`
- `generate_triton_tutorial.py`
"""


README = """# Triton 入门教程与 Excalidraw 图解

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
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagrams = [
        ("00_triton_learning_roadmap.excalidraw", roadmap()),
        ("01_triton_programming_model.excalidraw", programming_model()),
        ("02_triton_vector_add_softmax.excalidraw", vector_add_softmax()),
        ("03_triton_matmul_tile_flow.excalidraw", matmul_flow()),
        ("04_triton_autotune_debug_workflow.excalidraw", autotune_debug()),
        ("05_triton_interview_map.excalidraw", interview_map()),
    ]
    for filename, diagram in diagrams:
        diagram.save(filename)
    (OUT_DIR / "TUTORIAL.md").write_text(TUTORIAL, encoding="utf-8")
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
