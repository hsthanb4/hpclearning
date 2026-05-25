from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUT_DIR.parent.parent
HTML_PATH = ROOT_DIR / "kv_cache_quantization_tutorial_2026.html"


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
    def __init__(self, title: str, width: int = 1700, height: int = 1180) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.elements: list[dict] = []
        self._counter = 0

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def _base(self, element_type: str, x: float, y: float, w: float, h: float, **style: object) -> dict:
        seed = 19000 + self._counter * 97
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
    ) -> dict:
        el = self._base(
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
        self.elements.append(el)
        return el

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
    ) -> dict:
        el = self._base(
            "text",
            x,
            y,
            w,
            h,
            strokeColor=color,
            backgroundColor="transparent",
            fillStyle="solid",
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
        return el

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
        self.text(text, x + 12, y + 10, w - 24, h - 20, size=size, color=color)

    def label(self, text: str, x: float, y: float, w: float, h: float, color: str = PALETTE["title"]) -> None:
        self.text(text, x, y, w, h, size=22, color=color, align="left")

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = PALETTE["line"],
        stroke_style: str = "solid",
        end: str | None = "arrow",
    ) -> dict:
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
            fillStyle="solid",
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
        return el

    def title_block(self, subtitle: str) -> None:
        self.text(self.title, 80, 40, self.width - 160, 44, size=30, color=PALETTE["title"])
        self.text(subtitle, 80, 90, self.width - 160, 32, size=17, color=PALETTE["muted"])
        self.arrow(90, 132, self.width - 90, 132, "#94a3b8", "dashed", end=None)

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


def overview_diagram() -> Diagram:
    d = Diagram("KV Cache 量化总览")
    d.title_block("从内存公式、读写路径、量化轴到工程选型，一张图建立心智模型")

    d.label("1  生成链路", 90, 165, 260, 40)
    flow = [
        ("Prompt tokens\n输入序列", PALETTE["blue"]),
        ("Prefill\n并行算全量 Q K V", PALETTE["cyan"]),
        ("KV Cache\n逐层保存历史 K V", PALETTE["green"]),
        ("Decode step\n新 token 查询所有历史 K", PALETTE["purple"]),
        ("Output token\n追加到上下文", PALETTE["orange"]),
    ]
    x = 90
    for i, (text, fill) in enumerate(flow):
        d.box(text, x + i * 305, 220, 245, 105, fill, "#2563eb", size=18)
        if i < len(flow) - 1:
            d.arrow(x + i * 305 + 245, 272, x + (i + 1) * 305, 272)

    d.label("2  KV cache 内存公式", 90, 385, 360, 40)
    d.box(
        "KV bytes = 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_elem\n"
        "2 表示 K 和 V；GQA/MQA 主要通过减少 kv_heads 降低 cache 体积",
        90,
        435,
        690,
        110,
        PALETTE["yellow"],
        "#d97706",
        size=20,
    )
    d.box(
        "为什么 decode 容易 memory-bound\n每生成 1 个 token，都要读过去所有 token 的 K/V\n长上下文和大 batch 会让 HBM 读流量线性增长",
        830,
        435,
        720,
        110,
        PALETTE["red"],
        "#dc2626",
        size=20,
    )

    d.label("3  量化可以插入的位置", 90, 605, 420, 40)
    d.box("写入 cache 前\nK/V 从 FP16/BF16 转为低比特\n需要 scale、zero point 或 codebook", 95, 665, 360, 125, PALETTE["blue"], "#2563eb")
    d.box("存储形态\nFP8 / INT8 / INT4 / 2bit\npacked layout 决定真实压缩率", 500, 665, 360, 125, PALETTE["green"], "#16a34a")
    d.box("读取 cache 时\n反量化到计算域\n或在 FP8/INT8 域内直接 attention", 905, 665, 360, 125, PALETTE["purple"], "#7c3aed")
    d.box("残差窗口\n最近 token 保持高精度\n减少短程依赖损伤", 1310, 665, 285, 125, PALETTE["orange"], "#ea580c")
    d.arrow(455, 727, 500, 727)
    d.arrow(860, 727, 905, 727)
    d.arrow(1265, 727, 1310, 727)

    d.label("4  四个常见误区", 90, 855, 300, 40)
    pitfalls = [
        ("只看 bit 数\n忽略 scale 与 metadata 开销", PALETTE["slate"]),
        ("短上下文启用量化\n可能省显存但拖慢延迟", PALETTE["slate"]),
        ("把权重量化当 KV 量化\nAWQ/GPTQ 不自动压 KV", PALETTE["slate"]),
        ("只测困惑度\n长检索与推理任务更能暴露损伤", PALETTE["slate"]),
    ]
    for i, (text, fill) in enumerate(pitfalls):
        d.box(text, 95 + i * 390, 915, 330, 120, fill, "#64748b", size=18)
    d.box(
        "选型口诀\n先 FP8，再看是否真的被 KV 显存卡住；需要 4bit/3bit 时必须用目标模型、上下文长度、采样参数和业务任务做 A/B。",
        260,
        1075,
        1180,
        75,
        PALETTE["cyan"],
        "#0f766e",
        size=20,
    )
    return d


def turboquant_diagram() -> Diagram:
    d = Diagram("TurboQuant：论文算法与 vLLM 工程实现")
    d.title_block("不要把论文里的 QJL 残差修正和当前 vLLM 的 KV-cache preset 混为一谈")

    d.rect(70, 170, 750, 850, PALETTE["slate"], "#64748b", opacity=25)
    d.rect(880, 170, 750, 850, PALETTE["slate"], "#64748b", opacity=25)
    d.text("论文 TurboQuant", 100, 195, 690, 38, size=26, color=PALETTE["title"])
    d.text("vLLM TurboQuant KV-cache", 910, 195, 690, 38, size=26, color=PALETTE["title"])

    left_steps = [
        ("输入向量 x\n来自每个 head 的 K 或 V", PALETTE["blue"]),
        ("随机旋转 / Hadamard\n让坐标分布更集中、更接近独立", PALETTE["cyan"]),
        ("逐坐标 MSE 标量量化\n用少量 centroid 逼近旋转后坐标", PALETTE["green"]),
        ("残差 r = x - x_hat\nMSE 最优不等于内积无偏", PALETTE["orange"]),
        ("1-bit QJL 残差修正\n用符号 sketch 修正注意力内积偏差", PALETTE["purple"]),
        ("目标\n3.5 bit/channel 质量中性\n2.5 bit/channel 小幅退化", PALETTE["yellow"]),
    ]
    for i, (text, fill) in enumerate(left_steps):
        y = 260 + i * 115
        d.box(text, 135, y, 620, 82, fill, "#2563eb" if i < 3 else "#7c3aed", size=18)
        if i < len(left_steps) - 1:
            d.arrow(445, y + 82, 445, y + 115)

    d.box(
        "关键直觉\nAttention score 是 q dot k，K 的方向和内积误差最敏感。\n"
        "TurboQuant 不是普通 per-tensor affine int4，而是先改变几何分布再量化。",
        135,
        960,
        620,
        90,
        PALETTE["pink"],
        "#db2777",
        size=18,
    )

    right_steps = [
        ("命令入口\n--kv-cache-dtype turboquant_4bit_nc 等 preset", PALETTE["blue"]),
        ("Keys\nFP8 或 Hadamard + Lloyd-Max MSE 3/4 bit", PALETTE["cyan"]),
        ("Values\n3/4 bit uniform quantization", PALETTE["green"]),
        ("NC\nnorm correction 修正 centroid norm distortion", PALETTE["orange"]),
        ("当前实现取舍\nQJL 故意省略，避免 softmax 方差放大", PALETTE["red"]),
        ("工程结论\nFP8 默认更稳；4bit_nc 适合显存压力场景\nk3v4_nc 和 3bit_nc 要谨慎验证", PALETTE["yellow"]),
    ]
    for i, (text, fill) in enumerate(right_steps):
        y = 260 + i * 115
        d.box(text, 945, y, 620, 82, fill, "#2563eb" if i < 3 else "#dc2626", size=18)
        if i < len(right_steps) - 1:
            d.arrow(1255, y + 82, 1255, y + 115)

    d.arrow(790, 600, 910, 600, "#64748b", "dashed")
    d.text("同名不等于同实现\n读论文看算法保证\n跑服务看 kernel 和 preset", 805, 525, 220, 90, size=18, color=PALETTE["muted"])
    return d


def comparison_diagram() -> Diagram:
    d = Diagram("常用 KV Cache 量化方法对比与选型")
    d.title_block("按工程默认、研究低比特、极限长上下文三个层次理解")

    headers = ["方法", "核心做法", "常见比特", "优势", "风险和适用场景"]
    widths = [210, 410, 190, 360, 410]
    x0 = 65
    y0 = 180
    row_h = 120
    x_positions = [x0]
    for w in widths[:-1]:
        x_positions.append(x_positions[-1] + w)

    for i, header in enumerate(headers):
        d.box(header, x_positions[i], y0, widths[i], 58, PALETTE["blue"], "#1d4ed8", size=19, color="#1e3a8a")

    rows = [
        (
            "FP8 KV\nvLLM / TensorRT-LLM",
            "K/V 用 FP8 存储；scale 可 per-tensor 或 per-head；部分后端在 FP8 域算 attention",
            "8 bit",
            "部署最成熟；通常 2x cache 容量；延迟和吞吐更可预测",
            "压缩率有限；需要校准 scale 才更稳；先作为生产默认基线",
        ),
        (
            "HF QuantizedCache\nQuanto / HQQ",
            "Transformers generate 中量化 cache；可配 backend、axis、residual length",
            "int2 / int4 / int8",
            "上手快；适合单机实验和 OOM 缓解",
            "短上下文可能变慢；backend 轴设置不同；需要留高精度残差窗口",
        ),
        (
            "KIVI",
            "K per-channel，V per-token；利用 K 的 channel outlier 分布差异",
            "2 bit",
            "调参少；是低比特 KV 量化的经典基线",
            "工程实现依赖 kernel；长推理任务仍需业务验证",
        ),
        (
            "KVQuant",
            "Pre-RoPE K、non-uniform dtype、dense-sparse outlier、Q-Norm",
            "2 / 3 bit",
            "针对超长上下文；论文展示 1M 到 10M context 潜力",
            "实现复杂；需要自定义 CUDA；更像研究和专用系统路线",
        ),
        (
            "GEAR / SKVQ / WKVQuant",
            "量化叠加低秩误差、稀疏 outlier、滑动窗口或权重+KV 联合 PTQ",
            "约 1.5 到 4 bit",
            "针对更高压缩率或联合优化，提供很多设计思路",
            "方法各有额外假设；迁移到 vLLM/sglang 需要 kernel 和调度配合",
        ),
        (
            "TurboQuant",
            "旋转后逐坐标标量量化；论文加 QJL 残差修正；vLLM preset 加 norm correction",
            "3 / 4 bit\nk8v4",
            "在显存紧张时扩大可服务 token 容量；算法解释性强",
            "当前 vLLM 研究显示 FP8 仍是默认首选；4bit_nc 可试，3bit 需要谨慎",
        ),
    ]

    for r, row in enumerate(rows):
        y = y0 + 58 + r * row_h
        fill = PALETTE["slate"] if r % 2 == 0 else "#f8fafc"
        for c, cell in enumerate(row):
            d.box(cell, x_positions[c], y, widths[c], row_h, fill, "#cbd5e1", size=16)

    d.box(
        "实战决策树：先确认是不是 KV cache 限制吞吐或上下文；不是就别急着压 KV。是的话先测 FP8；仍 OOM 或排队严重，再比较 4bit / KIVI / TurboQuant / KVQuant。",
        160,
        980,
        1380,
        85,
        PALETTE["yellow"],
        "#d97706",
        size=20,
    )
    d.box(
        "评测必须覆盖：VRAM 峰值、TTFT、TPOT、throughput、cache hit、困惑度、长上下文检索、数学/代码推理、目标业务样本。",
        260,
        1090,
        1180,
        64,
        PALETTE["cyan"],
        "#0f766e",
        size=18,
    )
    return d


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TurboQuant 与 KV Cache 量化方法详细教程</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --surface: #ffffff;
      --surface-2: #eef4ff;
      --ink: #172033;
      --muted: #5b667a;
      --line: #d7deea;
      --blue: #1d4ed8;
      --green: #15803d;
      --orange: #c2410c;
      --purple: #6d28d9;
      --red: #b91c1c;
      --teal: #0f766e;
      --code-bg: #101828;
      --code-ink: #e6edf7;
      --radius: 8px;
      --shadow: 0 14px 38px rgba(31, 41, 55, 0.10);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Segoe UI", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
      line-height: 1.6;
    }

    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255, 255, 255, 0.95);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }

    .nav {
      max-width: 1220px;
      margin: 0 auto;
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      font-weight: 760;
      color: var(--ink);
      white-space: nowrap;
    }

    .links {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .links a {
      font-size: 14px;
      color: var(--muted);
      padding: 6px 9px;
      border-radius: 6px;
    }

    .links a:hover {
      background: #edf2ff;
      color: var(--blue);
      text-decoration: none;
    }

    main {
      max-width: 1220px;
      margin: 0 auto;
      padding: 34px 20px 70px;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.18fr) minmax(300px, 0.82fr);
      gap: 24px;
      align-items: stretch;
      margin-bottom: 26px;
    }

    .hero-copy, .hero-side, section {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }

    .hero-copy { padding: 30px; }

    h1 {
      font-size: clamp(32px, 5vw, 54px);
      line-height: 1.06;
      margin: 0 0 18px;
      letter-spacing: 0;
    }

    .lead {
      color: var(--muted);
      font-size: 18px;
      max-width: 760px;
      margin: 0 0 18px;
    }

    .stamp {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: #1e40af;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 14px;
      font-weight: 700;
    }

    .hero-side {
      background: #101828;
      color: #fff;
      padding: 24px;
      display: grid;
      align-content: center;
      gap: 14px;
    }

    .metric {
      display: grid;
      grid-template-columns: 105px 1fr;
      gap: 12px;
      align-items: baseline;
      padding: 12px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.16);
    }

    .metric:last-child { border-bottom: 0; }
    .metric b { font-size: 24px; color: #bfdbfe; }
    .metric span { color: #dbeafe; }

    section {
      margin: 24px 0;
      padding: 24px;
    }

    h2 {
      margin: 0 0 16px;
      font-size: 26px;
      line-height: 1.2;
    }

    h3 {
      margin: 18px 0 10px;
      font-size: 20px;
    }

    p { margin: 0 0 12px; }

    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }

    .grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }

    .callout {
      border-radius: var(--radius);
      padding: 18px;
      border: 1px solid var(--line);
      background: #f8fafc;
    }

    .callout.blue { background: #eff6ff; border-color: #bfdbfe; }
    .callout.green { background: #f0fdf4; border-color: #bbf7d0; }
    .callout.orange { background: #fff7ed; border-color: #fed7aa; }
    .callout.purple { background: #f5f3ff; border-color: #ddd6fe; }
    .callout.red { background: #fef2f2; border-color: #fecaca; }

    .label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 4px 9px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 760;
      color: #fff;
      background: var(--blue);
      margin-bottom: 10px;
    }

    .label.green { background: var(--green); }
    .label.orange { background: var(--orange); }
    .label.purple { background: var(--purple); }
    .label.red { background: var(--red); }
    .label.teal { background: var(--teal); }

    table {
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      display: table;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 12px;
      text-align: left;
      vertical-align: top;
      font-size: 15px;
    }

    th { background: #eef4ff; color: #1e3a8a; }
    tr:last-child td { border-bottom: 0; }

    .diagram-shell {
      margin: 18px 0 8px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
      background: #f8fafc;
    }

    .diagram-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      font-weight: 760;
    }

    .diagram-title a {
      font-weight: 700;
      font-size: 14px;
    }

    .kv-flow {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 12px;
      padding: 18px;
    }

    .flow-cell {
      min-height: 112px;
      border-radius: var(--radius);
      border: 1px solid #cbd5e1;
      padding: 14px;
      background: #fff;
      display: grid;
      align-content: center;
      position: relative;
    }

    .flow-cell b {
      display: block;
      color: #1e40af;
      margin-bottom: 8px;
    }

    .flow-cell:not(:last-child)::after {
      content: "";
      position: absolute;
      right: -13px;
      top: 50%;
      width: 13px;
      border-top: 2px solid #2563eb;
    }

    .quant-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      padding: 18px;
    }

    .pipeline {
      display: grid;
      gap: 10px;
    }

    .step {
      border-radius: var(--radius);
      border: 1px solid #cbd5e1;
      padding: 12px 14px;
      background: #fff;
    }

    .step b { color: #1e40af; }

    pre {
      background: var(--code-bg);
      color: var(--code-ink);
      border-radius: var(--radius);
      padding: 16px;
      overflow-x: auto;
      line-height: 1.55;
      font-size: 14px;
      margin: 12px 0 0;
    }

    code {
      font-family: "Cascadia Code", Consolas, Menlo, monospace;
      font-size: 0.94em;
    }

    .method-card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: #fff;
      padding: 16px;
    }

    .method-card h3 {
      margin-top: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .tag {
      flex: 0 0 auto;
      border-radius: 999px;
      background: #e0f2fe;
      color: #075985;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 760;
    }

    .checklist {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .checklist li {
      border-left: 4px solid #2563eb;
      background: #f8fafc;
      padding: 10px 12px;
      border-radius: 6px;
    }

    .source-list {
      columns: 2;
      column-gap: 32px;
    }

    .source-list li {
      break-inside: avoid;
      margin: 0 0 8px;
    }

    @media (max-width: 900px) {
      .hero, .grid-2, .grid-3, .quant-grid { grid-template-columns: 1fr; }
      .nav { align-items: flex-start; flex-direction: column; }
      .links { justify-content: flex-start; }
      .kv-flow { grid-template-columns: 1fr; }
      .flow-cell:not(:last-child)::after { display: none; }
      .source-list { columns: 1; }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <nav class="nav">
      <a class="brand" href="#top">KV Cache Quant Tutorial</a>
      <div class="links">
        <a href="#mental-model">心智模型</a>
        <a href="#turboquant">TurboQuant</a>
        <a href="#methods">方法对比</a>
        <a href="#practice">工程实践</a>
        <a href="#benchmark">评测</a>
        <a href="#sources">资料</a>
      </div>
    </nav>
  </header>

  <main id="top">
    <div class="hero">
      <div class="hero-copy">
        <span class="stamp">更新日期：2026-05-25</span>
        <h1>TurboQuant 与常用 KV Cache 量化方法详细教程</h1>
        <p class="lead">这份教程把 KV cache 量化拆成三个层次：先理解 cache 为什么吃显存，再理解量化轴和误差来源，最后把 TurboQuant、FP8、KIVI、KVQuant、GEAR、HF QuantizedCache、TensorRT-LLM 和 vLLM 工程实现放到同一张选型地图里。</p>
        <p class="lead">配套 Excalidraw 图稿可直接打开编辑，适合录课、面试复盘或继续扩展成技术分享。</p>
      </div>
      <aside class="hero-side">
        <div class="metric"><b>2x</b><span>FP8 KV cache 的典型容量提升，是生产默认基线。</span></div>
        <div class="metric"><b>2bit</b><span>KIVI 经典目标，K per-channel 与 V per-token 是关键。</span></div>
        <div class="metric"><b>3-4bit</b><span>TurboQuant 和 KVQuant 关注的低比特区间，必须做任务级验证。</span></div>
        <div class="metric"><b>1M+</b><span>KVQuant、SKVQ 等论文面向超长上下文的主要卖点。</span></div>
      </aside>
    </div>

    <section id="mental-model">
      <h2>1. 先建立 KV cache 的内存和执行模型</h2>
      <p>Transformer decode 阶段每次只生成一个新 token，但这个 token 的 query 需要和所有历史 token 的 key 做 attention score，再用 score 加权所有历史 value。KV cache 的意义是避免重复计算历史 token 的 K/V；代价是历史越长、并发越高，显存和 HBM 读流量越大。</p>
      <div class="diagram-shell">
        <div class="diagram-title">
          <span>图 1：KV cache 量化总览</span>
          <a href="excalidraw/kvcache_quant_tutorial_2026/00_kvcache_quant_overview.excalidraw">打开 Excalidraw 源文件</a>
        </div>
        <div class="kv-flow" aria-label="KV cache flow diagram">
          <div class="flow-cell"><b>Prompt</b><span>输入 token 进入模型。</span></div>
          <div class="flow-cell"><b>Prefill</b><span>并行计算整段上下文的 Q/K/V。</span></div>
          <div class="flow-cell"><b>KV Cache</b><span>逐层保存历史 K/V，供后续 token 复用。</span></div>
          <div class="flow-cell"><b>Decode</b><span>每步读取历史 K/V，读流量随上下文线性增加。</span></div>
          <div class="flow-cell"><b>Quantize</b><span>写入前压缩，读取时反量化或直接量化域计算。</span></div>
        </div>
      </div>

      <div class="grid-2">
        <div class="callout blue">
          <span class="label">内存公式</span>
          <p><code>KV bytes = 2 * layers * batch * tokens * kv_heads * head_dim * bytes_per_elem</code>。其中 2 是 K 和 V；GQA/MQA 通过减少 <code>kv_heads</code> 降低 cache 体积；FP16/BF16 的 <code>bytes_per_elem</code> 是 2，FP8 是 1，4bit 理论上是 0.5。</p>
        </div>
        <div class="callout orange">
          <span class="label orange">真实压缩率</span>
          <p>低比特不等于理论比例。scale、zero point、codebook、残差窗口、packing 对齐、block metadata 都会吃掉一部分收益。工程上必须看实际 VRAM 峰值和可服务 token 容量。</p>
        </div>
      </div>
    </section>

    <section>
      <h2>2. KV cache 量化的核心变量</h2>
      <table>
        <thead>
          <tr>
            <th>变量</th>
            <th>常见选择</th>
            <th>为什么重要</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>量化对象</td>
            <td>K、V，或者 Q/K/V 一起</td>
            <td>K 影响 attention score，误差直接改变路由；V 影响加权输出。vLLM 的 FP8 路径在部分后端也会量化 Q 并在 FP8 域执行 attention。</td>
          </tr>
          <tr>
            <td>量化轴</td>
            <td>per-tensor、per-head、per-token、per-channel、group-wise</td>
            <td>K 通常有 channel outlier，V 更适合 per-token。KIVI 的核心就是这个不对称观察。</td>
          </tr>
          <tr>
            <td>码型</td>
            <td>FP8、INT8、INT4、INT2、non-uniform、Lloyd-Max centroid</td>
            <td>FP8 工程成熟；INT4/INT2 压缩强但需要更细的 scale、outlier 或旋转处理。</td>
          </tr>
          <tr>
            <td>高精度残差</td>
            <td>recent window、first tokens、outlier sparse matrix、QJL residual</td>
            <td>最近 token、特殊 token、outlier 对输出质量影响大，很多方案不会把所有 token 一刀切压到低比特。</td>
          </tr>
          <tr>
            <td>计算路径</td>
            <td>读取后反量化，或在量化域执行 attention</td>
            <td>省显存不一定省延迟。反量化和 packing/unpacking 可能抵消收益；只有 kernel 配好，才会提高吞吐。</td>
          </tr>
        </tbody>
      </table>
    </section>

    <section id="turboquant">
      <h2>3. TurboQuant 讲清楚：论文算法 vs vLLM 实现</h2>
      <p>TurboQuant 论文的目标不是简单把每个数线性映射到 INT4，而是把高维向量先随机旋转，让坐标分布更容易被逐坐标标量量化；然后针对内积估计的偏差，用 1-bit QJL 对残差做修正。论文报告 KV cache 在 3.5 bit/channel 达到质量中性，在 2.5 bit/channel 只有小幅退化。</p>
      <div class="diagram-shell">
        <div class="diagram-title">
          <span>图 2：TurboQuant 论文算法与 vLLM 工程实现</span>
          <a href="excalidraw/kvcache_quant_tutorial_2026/01_turboquant_pipeline.excalidraw">打开 Excalidraw 源文件</a>
        </div>
        <div class="quant-grid">
          <div class="pipeline">
            <div class="step"><b>论文路径 1：</b>随机旋转或 Hadamard 旋转，把任意输入向量转到更容易量化的坐标系。</div>
            <div class="step"><b>论文路径 2：</b>对旋转后坐标做 MSE 最优标量量化，获得低比特近似。</div>
            <div class="step"><b>论文路径 3：</b>MSE 最优会带来 inner product bias，因此用 1-bit QJL sketch 修正残差。</div>
          </div>
          <div class="pipeline">
            <div class="step"><b>vLLM 路径 1：</b>提供 <code>turboquant_k8v4</code>、<code>turboquant_4bit_nc</code>、<code>turboquant_k3v4_nc</code>、<code>turboquant_3bit_nc</code> 等 preset。</div>
            <div class="step"><b>vLLM 路径 2：</b>keys 使用 FP8 或 Hadamard + Lloyd-Max MSE 量化；values 使用 3/4bit uniform 量化。</div>
            <div class="step"><b>vLLM 路径 3：</b>当前文档说明 QJL 被省略，原因是工程社区观察到它可能通过 softmax 放大方差。</div>
          </div>
        </div>
      </div>

      <div class="grid-3">
        <div class="callout purple">
          <span class="label purple">适合讲法</span>
          <p>TurboQuant 的本质是“先让向量几何变得好量化，再把内积误差作为一等目标处理”。这比“把 FP16 切成 INT4”更准确。</p>
        </div>
        <div class="callout green">
          <span class="label green">工程建议</span>
          <p>vLLM 2026-05 的研究结论是：FP8 仍是默认首选；TurboQuant <code>4bit_nc</code> 可以在显存压力极大时尝试。</p>
        </div>
        <div class="callout red">
          <span class="label red">不要误用</span>
          <p><code>k3v4_nc</code> 和 <code>3bit_nc</code> 在长上下文和推理任务上可能有明显质量损失，不能只看 cache 容量提升。</p>
        </div>
      </div>
    </section>

    <section id="methods">
      <h2>4. 常用 KV cache 量化方法对比</h2>
      <div class="diagram-shell">
        <div class="diagram-title">
          <span>图 3：方法对比与选型地图</span>
          <a href="excalidraw/kvcache_quant_tutorial_2026/02_kvcache_quant_methods_comparison.excalidraw">打开 Excalidraw 源文件</a>
        </div>
      </div>

      <div class="grid-2">
        <article class="method-card">
          <h3>FP8 KV Cache <span class="tag">生产默认</span></h3>
          <p>vLLM 支持 <code>kv_cache_dtype="fp8"</code>、<code>fp8_e4m3</code>、<code>fp8_e5m2</code>，scale 可默认、随机 token 校准或 dataset 校准。TensorRT-LLM 也支持 INT8/FP8 KV cache，并在 attention kernel 中读取时 on-the-fly dequant。</p>
          <p>优点是集成成熟、2x 容量提升清晰、性能更稳；缺点是压缩率不如 4bit/2bit。</p>
        </article>

        <article class="method-card">
          <h3>HF QuantizedCache <span class="tag">上手最快</span></h3>
          <p>Transformers 的 <code>cache_implementation="quantized"</code> 支持 <code>quanto</code> 和 <code>hqq</code> 后端。文档建议 HQQ 的 key/value axis 设为 1，Quanto 的 key/value axis 设为 0。</p>
          <p>它适合快速缓解单机 OOM，但短上下文可能因为量化和反量化开销变慢。</p>
        </article>

        <article class="method-card">
          <h3>KIVI <span class="tag">经典低比特</span></h3>
          <p>KIVI 的核心观察是 K cache 更适合 per-channel 量化，因为 key 存在 channel-wise outlier；V cache 更适合 per-token 量化。它主打 tuning-free 2bit KV cache，是很多后续方案的基线。</p>
        </article>

        <article class="method-card">
          <h3>KVQuant <span class="tag">超长上下文</span></h3>
          <p>KVQuant 组合了 per-channel key quantization、pre-RoPE key quantization、per-layer sensitivity non-uniform dtype、per-vector dense-and-sparse outlier handling 和 Q-Norm。它更像一条“为了 1M 到 10M context 专门设计系统和 kernel”的路线。</p>
        </article>

        <article class="method-card">
          <h3>GEAR / SKVQ / WKVQuant <span class="tag">研究路线</span></h3>
          <p>GEAR 先低比特量化大部分元素，再用低秩矩阵和稀疏矩阵补 quantization error。SKVQ 用 channel rearrange、clipped dynamic group quantization 和最近窗口高精度保留。WKVQuant 把权重和 KV cache 放在同一个 PTQ 框架中优化。</p>
        </article>

        <article class="method-card">
          <h3>TurboQuant <span class="tag">强压缩候选</span></h3>
          <p>论文版强调旋转、MSE 标量量化、QJL 残差修正；vLLM 版强调 Hadamard + Lloyd-Max、value uniform quantization 和 norm correction。工程使用时把 <code>turboquant_4bit_nc</code> 当作显存压力场景候选，而不是无脑默认。</p>
        </article>
      </div>
    </section>

    <section id="practice">
      <h2>5. 工程实践：从默认配置到低比特实验</h2>
      <h3>vLLM：先跑 FP8 baseline</h3>
      <pre><code># FP8 KV cache：生产上最值得先测的基线
vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-cache-dtype fp8

# 如果有校准数据，优先用 llm-compressor 生成更好的 KV scales
# vLLM 文档也支持 calculate_kv_scales=True 做 warmup 随机 token 校准</code></pre>

      <h3>vLLM：再测 TurboQuant preset</h3>
      <pre><code># 显存压力明显、排队严重时再试
vllm serve MiniMaxAI/MiniMax-M2.7 --kv-cache-dtype turboquant_4bit_nc

# 更激进的 3bit preset 只适合实验，必须看目标业务质量
vllm serve MiniMaxAI/MiniMax-M2.7 --kv-cache-dtype turboquant_3bit_nc</code></pre>

      <h3>Transformers：QuantizedCache 快速验证</h3>
      <pre><code>from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)

inputs = tokenizer("解释 KV cache quantization 的收益：", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=128,
    cache_implementation="quantized",
    cache_config={"backend": "quanto", "nbits": 4},
)
print(tokenizer.decode(out[0], skip_special_tokens=True))</code></pre>

      <div class="callout orange">
        <span class="label orange">实践提醒</span>
        <p>不要把 AWQ、GPTQ、bitsandbytes 这类权重量化当作 KV cache 量化。权重量化降低模型参数显存，KV cache 量化降低运行时上下文显存，两个瓶颈可能同时存在，也可能只有一个是瓶颈。</p>
      </div>
    </section>

    <section id="benchmark">
      <h2>6. 评测协议：不要只看能不能跑起来</h2>
      <ul class="checklist">
        <li><b>容量指标：</b>峰值 VRAM、可容纳最大 context、同一 GPU 上可承载的并发 batch、PagedAttention block 使用率。</li>
        <li><b>延迟指标：</b>TTFT、TPOT、p50/p95 latency、decode throughput、prefill throughput。</li>
        <li><b>质量指标：</b>困惑度只是烟雾测试；长上下文检索、needle、MRCR、数学推理、代码生成和业务样本更重要。</li>
        <li><b>对照设置：</b>BF16/FP16 baseline、FP8 baseline、目标低比特方案；固定模型、prompt 长度、输出长度、采样参数和硬件。</li>
        <li><b>失败定位：</b>K quantization 误差常体现为 attention routing 错；V quantization 误差常体现为输出表示偏。长上下文越长，低比特误差越容易累计。</li>
      </ul>
    </section>

    <section>
      <h2>7. 面试可用问答</h2>
      <div class="grid-2">
        <div class="callout blue">
          <span class="label">问：为什么 K 和 V 不该一样量化？</span>
          <p>K 参与 <code>q dot k</code>，决定 attention 分布；K 的 channel outlier 会直接扭曲路由。V 是被 score 加权求和，分布和敏感性不同。因此 KIVI 才会用 K per-channel、V per-token 的不对称设计，KV-AdaQuant 这类工作也强调 keys deserve more bits。</p>
        </div>
        <div class="callout green">
          <span class="label green">问：为什么 FP8 常是默认首选？</span>
          <p>因为它压缩率虽不极限，但 kernel、scale、硬件路径和质量风险更可控。vLLM 的 2026-05 TurboQuant 研究也建议大多数工作负载先选 FP8。</p>
        </div>
        <div class="callout purple">
          <span class="label purple">问：TurboQuant 的亮点是什么？</span>
          <p>它把高维向量量化问题转成“旋转后坐标分布更好处理”的问题，并把 attention 依赖的 inner product bias 显式作为目标，而不是只最小化每个数的误差。</p>
        </div>
        <div class="callout red">
          <span class="label red">问：为什么 3bit 不一定比 FP8 好？</span>
          <p>3bit 节省更多 cache，但可能需要 unpack、反量化、norm correction 或额外 kernel 逻辑；质量上也可能在长上下文和推理任务退化。最终要看服务吞吐和任务准确率的 Pareto frontier。</p>
        </div>
      </div>
    </section>

    <section id="sources">
      <h2>参考资料和配套文件</h2>
      <div class="grid-2">
        <div>
          <h3>配套 Excalidraw</h3>
          <ul>
            <li><a href="excalidraw/kvcache_quant_tutorial_2026/00_kvcache_quant_overview.excalidraw">00_kvcache_quant_overview.excalidraw</a></li>
            <li><a href="excalidraw/kvcache_quant_tutorial_2026/01_turboquant_pipeline.excalidraw">01_turboquant_pipeline.excalidraw</a></li>
            <li><a href="excalidraw/kvcache_quant_tutorial_2026/02_kvcache_quant_methods_comparison.excalidraw">02_kvcache_quant_methods_comparison.excalidraw</a></li>
          </ul>
        </div>
        <div>
          <h3>使用建议</h3>
          <p>HTML 用来系统阅读；Excalidraw 用来讲解。录视频时建议顺序是：图 1 讲瓶颈，图 2 讲 TurboQuant，图 3 讲工程选型。</p>
        </div>
      </div>
      <ol class="source-list">
        <li><a href="https://arxiv.org/abs/2504.19874">TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate</a></li>
        <li><a href="https://openreview.net/pdf?id=tO3ASKZlok">TurboQuant ICLR 2026 OpenReview PDF</a></li>
        <li><a href="https://vllm.ai/blog/2026-05-11-turboquant">vLLM: A First Comprehensive Study of TurboQuant</a></li>
        <li><a href="https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/">vLLM TurboQuant API docs</a></li>
        <li><a href="https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/">vLLM Quantized KV Cache docs</a></li>
        <li><a href="https://huggingface.co/docs/transformers/kv_cache">Hugging Face Transformers KV cache docs</a></li>
        <li><a href="https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html">TensorRT-LLM GPT Attention docs</a></li>
        <li><a href="https://arxiv.org/abs/2402.02750">KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache</a></li>
        <li><a href="https://arxiv.org/abs/2401.18079">KVQuant: Towards 10 Million Context Length LLM Inference</a></li>
        <li><a href="https://arxiv.org/abs/2403.05527">GEAR: Efficient KV Cache Compression</a></li>
        <li><a href="https://arxiv.org/abs/2405.06219">SKVQ: Sliding-window KV Cache Quantization</a></li>
        <li><a href="https://arxiv.org/abs/2402.12065">WKVQuant: Quantizing Weight and KV Cache</a></li>
        <li><a href="https://arxiv.org/abs/2502.15075">Quantize What Counts: More for Keys, Less for Values</a></li>
      </ol>
    </section>
  </main>
</body>
</html>
"""


README = """# KV Cache 量化教程：TurboQuant 与常用方法

更新时间：2026-05-25

本目录包含 3 张标准 `.excalidraw` 图稿，配合仓库根目录的 `kv_cache_quantization_tutorial_2026.html` 使用。

## 文件

1. `00_kvcache_quant_overview.excalidraw`：KV cache 内存公式、读写路径、量化插入点和常见误区。
2. `01_turboquant_pipeline.excalidraw`：TurboQuant 论文算法和 vLLM 工程实现的差异。
3. `02_kvcache_quant_methods_comparison.excalidraw`：FP8、HF QuantizedCache、KIVI、KVQuant、GEAR/SKVQ/WKVQuant、TurboQuant 的对比和选型。
4. `generate_kvcache_quant_tutorial.py`：重新生成 HTML 与 Excalidraw 的脚本。

## 主要资料源

- TurboQuant paper: https://arxiv.org/abs/2504.19874
- TurboQuant OpenReview PDF: https://openreview.net/pdf?id=tO3ASKZlok
- vLLM TurboQuant study: https://vllm.ai/blog/2026-05-11-turboquant
- vLLM quantized KV cache docs: https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/
- vLLM TurboQuant docs: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/
- Hugging Face KV cache docs: https://huggingface.co/docs/transformers/kv_cache
- TensorRT-LLM GPT attention docs: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
- KIVI: https://arxiv.org/abs/2402.02750
- KVQuant: https://arxiv.org/abs/2401.18079
- GEAR: https://arxiv.org/abs/2403.05527
- SKVQ: https://arxiv.org/abs/2405.06219
- WKVQuant: https://arxiv.org/abs/2402.12065
- Quantize What Counts: https://arxiv.org/abs/2502.15075
"""


def main() -> None:
    overview_diagram().save("00_kvcache_quant_overview.excalidraw")
    turboquant_diagram().save("01_turboquant_pipeline.excalidraw")
    comparison_diagram().save("02_kvcache_quant_methods_comparison.excalidraw")
    HTML_PATH.write_text(HTML, encoding="utf-8")
    (OUT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
