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
        seed = 28000 + self._counter * 137
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

    def title_block(self) -> None:
        self.text(self.title, 80, 34, self.width - 160, 42, size=30, color=PALETTE["title"])
        self.text(self.subtitle, 90, 83, self.width - 180, 34, size=17, color=PALETTE["muted"])
        self.arrow(90, 130, self.width - 90, 130, "#94a3b8", "dashed", end=None)

    def label(self, text: str, x: float, y: float, w: float = 360, h: float = 40) -> None:
        self.text(text, x, y, w, h, size=22, color=PALETTE["title"], align="left")

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


def learning_roadmap() -> Diagram:
    d = Diagram("TileLang 入门学习路线", "从能跑示例到能讲清 GPU kernel DSL 的完整路径")
    phases = [
        ("0  背景", "复习 CUDA block warp shared memory\n理解 TIR 和 tile 的关系", PALETTE["blue"]),
        ("1  安装", "pip 或源码安装\n确认 Python CUDA glibc 要求", PALETTE["cyan"]),
        ("2  读例子", "VectorAdd GEMM FlashAttention\n先看 program 再看 generated code", PALETTE["green"]),
        ("3  写 kernel", "用 T.Kernel 定义网格\n用 T.Parallel 和 memory scope 表达并行", PALETTE["yellow"]),
        ("4  调优", "tile size num threads pipeline stages\nautotune 加 profile 验证", PALETTE["orange"]),
        ("5  面试输出", "能解释 TileLang vs Triton CUDA TVM\n能画出 GEMM tile 数据流", PALETTE["purple"]),
    ]
    x = 85
    y = 205
    for i, (title, body, fill) in enumerate(phases):
        px = x + (i % 3) * 520
        py = y + (i // 3) * 310
        d.box(title, px, py, 390, 70, fill, "#2563eb", size=22)
        d.box(body, px, py + 90, 390, 140, "#ffffff", "#94a3b8", size=18)
        if i not in {2, 5}:
            d.arrow(px + 390, py + 35, px + 520, py + 35)
        if i == 2:
            d.arrow(px + 190, py + 230, px + 190, py + 310)
    d.label("每天怎么学", 90, 860)
    d.box("第一天  跑通安装与 VectorAdd\n第二天  拆 T.Kernel 和 T.Parallel\n第三天  手写 tiled GEMM 骨架", 90, 925, 470, 135, PALETTE["slate"], "#475569")
    d.box("第四天  加 shared memory 和 pipeline\n第五天  autotune 并看 profiler\n第六天  对比 Triton CUDA TVM", 615, 925, 470, 135, PALETTE["slate"], "#475569")
    d.box("第七天  复盘面试题\n目标不是背 API\n而是能解释编程模型和性能路径", 1140, 925, 470, 135, PALETTE["slate"], "#475569")
    return d


def mental_model() -> Diagram:
    d = Diagram("TileLang 编程模型总览", "Python DSL 前端到 TVM IR 再到可运行 kernel")
    d.label("用户写的部分", 85, 170)
    d.box("@tilelang.jit\nPython 函数返回 T.prim_func", 90, 230, 340, 100, PALETTE["blue"], "#2563eb")
    d.box("T.Tensor 和 T.Buffer\n描述形状 dtype 与输入输出", 90, 365, 340, 110, PALETTE["blue"], "#2563eb")
    d.box("T.Kernel grid threads\n描述 CTA 网格和线程块", 90, 510, 340, 110, PALETTE["blue"], "#2563eb")
    d.label("Tile 级 DSL", 555, 170)
    d.box("循环\nT.Parallel T.serial T.Pipelined T.unroll", 555, 230, 410, 100, PALETTE["green"], "#16a34a")
    d.box("存储层级\nglobal shared local fragment", 555, 365, 410, 110, PALETTE["green"], "#16a34a")
    d.box("算子原语\nT.copy T.gemm T.clear T.reduce_sum", 555, 510, 410, 110, PALETTE["green"], "#16a34a")
    d.label("编译和运行", 1090, 170)
    d.box("Lowering 到 TVM IR\n检查结构是否符合硬件映射", 1090, 230, 430, 100, PALETTE["purple"], "#7c3aed")
    d.box("生成目标代码\nCUDA HIP CPU 或其他后端", 1090, 365, 430, 110, PALETTE["purple"], "#7c3aed")
    d.box("Runtime 调用\n传入 torch tensor 并执行", 1090, 510, 430, 110, PALETTE["purple"], "#7c3aed")
    for y in [280, 420, 565]:
        d.arrow(430, y, 555, y)
        d.arrow(965, y, 1090, y)
    d.label("三层心智模型", 90, 700)
    d.box("Beginner\n用模板写出正确 kernel\n先让数据路径可解释", 110, 770, 400, 130, PALETTE["yellow"], "#d97706")
    d.box("Developer\n控制 tile 形状、线程绑定、共享内存、pipeline\n开始做性能调优", 635, 770, 400, 130, PALETTE["orange"], "#ea580c")
    d.box("Expert\n检查 IR、生成代码、tensorize、layout\n把 DSL 映射到具体 GPU 指令", 1160, 770, 400, 130, PALETTE["red"], "#dc2626")
    d.arrow(510, 835, 635, 835)
    d.arrow(1035, 835, 1160, 835)
    return d


def kernel_anatomy() -> Diagram:
    d = Diagram("TileLang Kernel 解剖图", "把一段 TileLang 程序拆成 shape、grid、loop、memory、primitive")
    d.label("程序骨架", 90, 170)
    d.box("def kernel(M, N, K, dtype):\n    @T.prim_func\n    def main(A, B, C):\n        with T.Kernel(grid, threads):", 95, 230, 520, 185, PALETTE["slate"], "#475569", size=18)
    d.box("形状和 dtype\n决定 buffer 维度\n决定 index 计算边界", 690, 230, 360, 105, PALETTE["blue"], "#2563eb")
    d.box("grid 和 threads\n决定 CTA 数量\n决定每个 CTA 的并行资源", 1150, 230, 360, 105, PALETTE["cyan"], "#0891b2")
    d.arrow(615, 295, 690, 295)
    d.arrow(1050, 295, 1150, 295)
    d.label("GPU 映射", 90, 480)
    d.box("Block 或 CTA\n处理一个 tile\n例如 C 的 BM x BN 子矩阵", 105, 545, 330, 125, PALETTE["green"], "#16a34a")
    d.box("Threads\nT.Parallel 负责并行索引\nthreadIdx 负责合作搬运或计算", 505, 545, 330, 125, PALETTE["green"], "#16a34a")
    d.box("Memory scopes\nglobal 到 shared 到 fragment\n减少 HBM 往返", 905, 545, 330, 125, PALETTE["green"], "#16a34a")
    d.box("Primitives\ncopy gemm reduce clear\n表达常见 tile 操作", 1305, 545, 300, 125, PALETTE["green"], "#16a34a")
    d.arrow(435, 607, 505, 607)
    d.arrow(835, 607, 905, 607)
    d.arrow(1235, 607, 1305, 607)
    d.label("写程序时按这个顺序检查", 90, 760)
    checklist = [
        ("1", "每个 CTA 负责哪块输出"),
        ("2", "输入 tile 从哪搬到 shared"),
        ("3", "K 维怎么循环和 pipeline"),
        ("4", "fragment 上怎么累加"),
        ("5", "边界和 dtype 是否正确"),
        ("6", "调优参数是否可枚举"),
    ]
    for i, (num, text) in enumerate(checklist):
        px = 110 + i * 250
        d.box(num, px, 835, 58, 58, PALETTE["yellow"], "#d97706", size=24)
        d.text(text, px - 35, 910, 130, 58, size=16, color=PALETTE["text"])
        if i < len(checklist) - 1:
            d.arrow(px + 58, 864, px + 250, 864)
    return d


def gemm_tile_flow() -> Diagram:
    d = Diagram("Tiled GEMM 数据流", "用一张图理解 TileLang 为什么要显式表达 tile、shared memory 和 pipeline")
    d.label("全局内存", 90, 170)
    d.box("A matrix\nM x K", 110, 245, 270, 90, PALETTE["blue"], "#2563eb", size=22)
    d.box("B matrix\nK x N", 110, 395, 270, 90, PALETTE["blue"], "#2563eb", size=22)
    d.box("C matrix\nM x N", 110, 545, 270, 90, PALETTE["blue"], "#2563eb", size=22)
    d.label("一个 CTA 的工作", 520, 170)
    d.rect(525, 240, 530, 420, "#ffffff", "#475569")
    d.box("A tile\nBM x BK\nshared", 565, 285, 180, 100, PALETTE["cyan"], "#0891b2")
    d.box("B tile\nBK x BN\nshared", 815, 285, 180, 100, PALETTE["cyan"], "#0891b2")
    d.box("Accumulator\nBM x BN\nfragment", 675, 485, 240, 105, PALETTE["yellow"], "#d97706")
    d.arrow(655, 385, 735, 485)
    d.arrow(905, 385, 855, 485)
    d.label("K 维主循环", 1195, 170)
    d.box("for ko in K tiles\nload A tile and B tile", 1195, 245, 360, 95, PALETTE["green"], "#16a34a")
    d.box("pipeline stages\n当前 tile 计算\n下一 tile 预取", 1195, 385, 360, 110, PALETTE["orange"], "#ea580c")
    d.box("store C tile\n写回 global", 1195, 545, 360, 90, PALETTE["purple"], "#7c3aed")
    d.arrow(380, 290, 565, 335)
    d.arrow(380, 440, 815, 335)
    d.arrow(915, 537, 1195, 590)
    d.arrow(1375, 340, 1375, 385)
    d.arrow(1375, 495, 1375, 545)
    d.label("性能要点", 90, 740)
    d.box("tile size 太小\n访存复用不足\nlaunch 和同步开销占比高", 110, 805, 330, 125, PALETTE["red"], "#dc2626")
    d.box("tile size 太大\nshared memory 或 register 压力升高\noccupancy 下降", 500, 805, 330, 125, PALETTE["red"], "#dc2626")
    d.box("pipeline 合理\n隐藏 global memory 延迟\n提高 tensor core 利用", 890, 805, 330, 125, PALETTE["green"], "#16a34a")
    d.box("边界处理清晰\n非整除 shape 也能跑\n先正确再调优", 1280, 805, 330, 125, PALETTE["yellow"], "#d97706")
    return d


def tuning_debug_workflow() -> Diagram:
    d = Diagram("TileLang 调优与调试工作流", "从正确性到性能，再回到 IR 和 profiler 定位瓶颈")
    steps = [
        ("写 baseline", "先写最直观 tile kernel\n用小 shape 对齐 torch 结果", PALETTE["blue"]),
        ("打印和检查", "用 T.print 或生成代码\n确认索引和边界", PALETTE["cyan"]),
        ("加 pipeline", "T.Pipelined 控制阶段\n检查 shared memory 生命周期", PALETTE["green"]),
        ("autotune", "枚举 block size num threads\n记录 latency 和 best config", PALETTE["yellow"]),
        ("profile", "Nsight 看 occupancy memory tensor core\n不要只看单次耗时", PALETTE["orange"]),
        ("固化模板", "把 shape family 和配置记录下来\n形成可复用 kernel", PALETTE["purple"]),
    ]
    for i, (title, body, fill) in enumerate(steps):
        px = 100 + (i % 3) * 520
        py = 220 + (i // 3) * 305
        d.box(title, px, py, 370, 70, fill, "#2563eb", size=22)
        d.box(body, px, py + 86, 370, 130, "#ffffff", "#94a3b8", size=18)
        if i not in {2, 5}:
            d.arrow(px + 370, py + 35, px + 520, py + 35)
        if i == 2:
            d.arrow(px + 185, py + 216, px + 185, py + 305)
    d.label("常见失败模式", 90, 875)
    d.box("结果错误\n索引越界、mask 缺失、dtype 累加错误", 110, 940, 360, 110, PALETTE["red"], "#dc2626")
    d.box("性能不涨\ntile 形状不匹配硬件、访存未合并、pipeline 没隐藏延迟", 505, 940, 430, 110, PALETTE["red"], "#dc2626")
    d.box("只会调参数\n没有先画数据流\n不知道瓶颈在 HBM shared 还是 register", 970, 940, 430, 110, PALETTE["red"], "#dc2626")
    return d


def interview_map() -> Diagram:
    d = Diagram("TileLang 面试速查图", "把学习内容收束成可讲清楚的问答框架")
    d.label("一句话定义", 90, 170)
    d.box("TileLang 是面向高性能算子的 Python DSL\n核心是用 tile 级抽象表达 GPU kernel\n再经由 TVM 体系 lowering 到目标代码", 105, 230, 660, 130, PALETTE["blue"], "#2563eb", size=20)
    d.label("和其他工具比较", 900, 170)
    d.box("vs CUDA\nCUDA 控制最细 但样板多\nTileLang 更接近算法 tile 表达", 900, 230, 310, 135, PALETTE["yellow"], "#d97706")
    d.box("vs Triton\nTriton 编程体验更成熟\nTileLang 更强调 TVM IR 和调度体系衔接", 1240, 230, 330, 135, PALETTE["yellow"], "#d97706")
    d.label("必须能画出的图", 90, 430)
    d.box("GEMM tile\nA/B global 到 shared\nfragment 累加\nC 写回", 105, 495, 300, 120, PALETTE["green"], "#16a34a")
    d.box("FlashAttention tile\nQ block 扫 K/V tile\nonline softmax 更新 m l O", 465, 495, 350, 120, PALETTE["green"], "#16a34a")
    d.box("调优循环\nbaseline correctness\nautotune profile 固化", 875, 495, 330, 120, PALETTE["green"], "#16a34a")
    d.box("编译路径\nPython DSL 到 TIR\n到 CUDA source\n到 runtime", 1265, 495, 330, 120, PALETTE["green"], "#16a34a")
    d.label("常见追问", 90, 690)
    d.box("为什么要 shared memory\n减少 global memory 访问\n提升 tile 复用", 105, 755, 330, 115, PALETTE["purple"], "#7c3aed")
    d.box("pipeline 解决什么\n隐藏 global load 延迟\n让搬运和计算重叠", 475, 755, 330, 115, PALETTE["purple"], "#7c3aed")
    d.box("autotune 调什么\ntile size threads stages\n目标是延迟和资源平衡", 845, 755, 330, 115, PALETTE["purple"], "#7c3aed")
    d.box("怎么验证\n小 shape 对齐 torch\n大 shape profile\n再看生成代码", 1215, 755, 330, 115, PALETTE["purple"], "#7c3aed")
    d.label("回答模板", 90, 940)
    d.box("先讲算法 tile\n再讲 GPU 映射\n最后讲调优和验证\n不要一上来背 API", 105, 990, 1450, 95, PALETTE["slate"], "#475569", size=22)
    return d


TUTORIAL = """# TileLang 入门详细教程

更新时间：2026-05-25

这份教程面向已经学过 CUDA、Triton、MLIR 或高性能算子的人。目标不是背 API，而是建立一个能写、能调、能在面试里讲清楚的 TileLang 心智模型。

资料核验：

- TileLang docs: https://tilelang.com/
- TileLang GitHub: https://github.com/tile-ai/tilelang
- TileLang releases: https://github.com/tile-ai/tilelang/releases
- TileLang paper: https://arxiv.org/abs/2504.17577
- TVM docs: https://tvm.apache.org/docs/

截至 2026-05-25，TileLang 文档站页面显示版本为 0.1.10，GitHub releases 页面最新稳定标签为 v0.1.9。学习时把文档版本和 release 标签分开看。

## 1. TileLang 是什么

TileLang 是一个面向高性能 kernel 的 Python DSL。你用 Python 写 tile 级别的程序结构，TileLang 把它 lowering 到 TVM IR，再生成目标后端代码并运行。

用一句话记：

> TileLang 让你用接近算法 tile 的方式描述 GPU kernel，同时保留 shared memory、thread、pipeline、tensor core 等性能控制点。

它适合学习和实现这些东西：

- GEMM、conv、reduction、elementwise fusion。
- FlashAttention、linear attention、MoE、quantization kernel。
- 需要从算法 tile 推到 GPU memory hierarchy 的自定义算子。
- 想理解 TVM/TIR 但又不想一开始就被 schedule API 淹没的人。

## 2. 先学什么

建议顺序：

1. 复习 CUDA block、warp、thread、shared memory、register、coalesced load。
2. 跑通 TileLang 安装和官方 examples。
3. 先读 VectorAdd，再读 GEMM，再读 FlashAttention 类例子。
4. 自己写一个 tiled GEMM 骨架。
5. 加 pipeline、autotune、profile。
6. 回头看生成的 IR 和 CUDA 代码。

对应图：`00_tilelang_learning_roadmap.excalidraw`

## 3. 安装和环境

官方文档给出的关键要求包括：

- Python 3.10 或更高。
- Linux glibc 2.28 或更高。
- CUDA 要求分两类：宿主机 CUDA 10 或更高，或者 pip 提供的 CUDA toolchain 13 或更高。
- Windows 源码构建需要 Python、CMake、Visual Studio Build Tools 和 MSVC C++ toolchain。

常见安装路径：

```bash
pip install tilelang
pip install git+https://github.com/tile-ai/tilelang.git
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install . -v
python -c "import tilelang; print(tilelang.__version__)"
```

本仓库当前没有把 TileLang 作为依赖安装；这份教程提供学习路径和图解，不声明本机已经成功运行 TileLang kernel。

## 4. 编程模型

TileLang 程序通常分三层理解：

- Python 函数层：用 `@tilelang.jit` 包住一个返回 `T.prim_func` 的函数。
- Tile DSL 层：用 `T.Kernel`、`T.Parallel`、`T.copy`、`T.gemm`、memory scope 表达 tile 级计算。
- 编译运行层：TileLang lowering 到 TVM IR，生成目标代码，再绑定运行时。

对应图：`01_tilelang_programming_model.excalidraw`

你读一个 TileLang kernel 时，先找这五个东西：

1. 输入输出 tensor 的 shape 和 dtype。
2. `T.Kernel` 的 grid 和 threads。
3. 每个 CTA 负责哪一块输出 tile。
4. global、shared、local、fragment 的数据搬运路径。
5. 计算 primitive 和 loop pipeline。

## 5. 最小 VectorAdd 思路

VectorAdd 的学习价值不是性能，而是确认 DSL 的基本形状：

```python
import tilelang
import tilelang.language as T
from tilelang import jit


@jit
def vector_add(n, block_size=256, dtype="float32"):
    @T.prim_func
    def main(a: T.Tensor((n,), dtype),
             b: T.Tensor((n,), dtype),
             c: T.Tensor((n,), dtype)):
        with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                i = bx * block_size + tx
                if i < n:
                    c[i] = a[i] + b[i]
    return main


kernel = vector_add(1024)
```

这段代码的重点：

- `T.Kernel` 建立 block 维度。
- `thread_binding` 建立 thread 维度。
- `i < n` 是边界保护。
- 先跑小 shape 对齐 torch，再扩到大 shape。

## 6. GEMM 怎么学

GEMM 是 TileLang 入门的核心练习，因为它把所有重要概念都串起来：

- A 和 B 从 global memory 搬到 shared memory。
- 每个 CTA 负责 C 的一个 BM x BN tile。
- K 维按 BK 切块循环。
- fragment accumulator 累加。
- pipeline 尝试隐藏 global load 延迟。
- autotune 搜索 BM、BN、BK、threads、stages。

对应图：`03_tilelang_gemm_tile_flow.excalidraw`

写 GEMM 时先画数据流，再写代码。不要先陷入 API 细节：

1. 画出 C tile。
2. 反推这个 C tile 需要哪块 A tile 和 B tile。
3. 决定 A/B tile 怎么搬到 shared。
4. 决定 K loop 怎么展开或 pipeline。
5. 决定 accumulator dtype。
6. 最后才调 tile size。

## 7. 调优与调试

TileLang 的调优不应该从随机改参数开始。推荐 workflow：

1. 写 baseline，只求正确。
2. 用小 shape 和 PyTorch 结果对齐。
3. 打印或检查生成代码，确认索引和边界。
4. 加 shared memory 复用。
5. 加 pipeline。
6. 用 autotune 枚举关键参数。
7. 用 profiler 看瓶颈。
8. 固化 shape family 和最佳配置。

对应图：`04_tilelang_tuning_debug_workflow.excalidraw`

常见坑：

- tile size 太大导致 register 或 shared memory 压力过高。
- tile size 太小导致访存复用不足。
- pipeline stage 加了，但 load 和 compute 没真正重叠。
- 只看 latency，不看 occupancy、memory throughput、tensor core utilization。
- 边界 shape 没测，只测了能整除的矩阵。

## 8. 和 CUDA、Triton、TVM 怎么比较

面试里可以这样讲：

- CUDA：控制最细，工程样板最多。适合极限手工优化。
- Triton：Python 体验成熟，面向 block program，常用于 PyTorch 自定义算子。
- TVM：编译体系完整，schedule 和后端覆盖广，但学习曲线更陡。
- TileLang：把 tile 级 DSL、TVM IR、GPU kernel 控制点放在一起，适合用较高层表达写出可调优 kernel。

不要说 TileLang 一定替代 Triton 或 CUDA。更稳的说法是：

> TileLang 适合在自定义高性能 kernel 中更清楚地表达 tile 数据流，并利用 TVM 编译基础设施做 lowering、生成和调优。

## 9. 七天练习计划

第一天：安装、跑 VectorAdd、理解 `T.Kernel`。

第二天：读 GEMM example，把 block、thread、shared、fragment 标出来。

第三天：自己写一个最小 tiled GEMM 骨架，不追求快。

第四天：加 shared memory staging，比较 global-only 与 shared 版本。

第五天：加入 pipeline，观察代码结构变化。

第六天：autotune tile 参数，记录不同 shape 的 best config。

第七天：用 `05_tilelang_interview_map.excalidraw` 复盘面试题。

## 10. 面试问答

Q：TileLang 的核心抽象是什么？

A：用 tile 级 DSL 描述 kernel 数据流和调度意图，包括 grid、thread、memory scope、copy、gemm、pipeline，再经由 TVM IR lowering 到目标代码。

Q：为什么要先画 GEMM tile 数据流？

A：因为性能来自数据复用和硬件映射。你必须先知道每个 CTA 算哪块 C、需要哪块 A/B、怎么进入 shared、在哪个 fragment 累加，才能合理调参数。

Q：autotune 调什么？

A：主要调 tile shape、threads、pipeline stages、memory layout 等参数。目标是在 occupancy、register pressure、shared memory、memory bandwidth、tensor core 利用之间平衡。

Q：什么时候不用 TileLang？

A：如果只需要一个很普通的 PyTorch op，就不需要自定义 kernel。如果团队已经有成熟 Triton/CUDA 模板，并且 TileLang 后端或部署环境不匹配，也不应强行迁移。

## 11. 文件清单

- `00_tilelang_learning_roadmap.excalidraw`：学习路线。
- `01_tilelang_programming_model.excalidraw`：编程模型。
- `02_tilelang_kernel_anatomy.excalidraw`：kernel 解剖。
- `03_tilelang_gemm_tile_flow.excalidraw`：GEMM tile 数据流。
- `04_tilelang_tuning_debug_workflow.excalidraw`：调优与调试。
- `05_tilelang_interview_map.excalidraw`：面试速查。
- `generate_tilelang_tutorial.py`：重新生成本目录所有教程文件。
"""


README = """# TileLang 入门教程与 Excalidraw 图解

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
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagrams = [
        ("00_tilelang_learning_roadmap.excalidraw", learning_roadmap()),
        ("01_tilelang_programming_model.excalidraw", mental_model()),
        ("02_tilelang_kernel_anatomy.excalidraw", kernel_anatomy()),
        ("03_tilelang_gemm_tile_flow.excalidraw", gemm_tile_flow()),
        ("04_tilelang_tuning_debug_workflow.excalidraw", tuning_debug_workflow()),
        ("05_tilelang_interview_map.excalidraw", interview_map()),
    ]
    for filename, diagram in diagrams:
        diagram.save(filename)
    (OUT_DIR / "TUTORIAL.md").write_text(TUTORIAL, encoding="utf-8")
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
