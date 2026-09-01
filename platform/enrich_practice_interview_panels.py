from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PANEL_PREFIX = "practice-interview-panel-"


PALETTE = {
    "title": "#1e40af",
    "text": "#243042",
    "muted": "#64748b",
    "blue": "#bfdbfe",
    "green": "#bbf7d0",
    "yellow": "#fef3c7",
    "purple": "#ddd6fe",
    "slate": "#e2e8f0",
}


TOPIC_RULES: list[tuple[str, str, list[str], list[str]]] = [
    (
        "triton",
        "Triton GPU kernel",
        [
            "用 VectorAdd 改成 axpy, 比较 BLOCK_SIZE 对带宽的影响",
            "实现 row-wise softmax, 处理非 2 的幂列数和 mask",
            "画出 matmul 的 program_id 到 C tile 的映射",
        ],
        [
            "Triton program 和 CUDA thread/block 的抽象差异是什么",
            "tl.constexpr、mask、tl.arange 分别解决什么问题",
            "matmul 中 GROUP_M 为什么能改善 L2 cache 复用",
        ],
    ),
    (
        "tilelang",
        "TileLang tiled kernel DSL",
        [
            "把同一个 GEMM tile 用 T.Kernel/T.copy/T.gemm 重新讲一遍",
            "修改 BM/BN/BK 并记录 shared memory 与寄存器压力变化",
            "用小 shape 对齐 torch baseline, 再检查生成代码",
        ],
        [
            "TileLang 和 Triton 的抽象边界有什么不同",
            "T.Pipelined 与 shared memory staging 解决什么瓶颈",
            "为什么先画 tile 数据流再写 DSL 代码",
        ],
    ),
    (
        "kvcache",
        "KV cache 量化",
        [
            "计算不同 batch/context/kv_heads 下的 KV cache 显存",
            "对比 FP8、INT4、2bit 在显存、质量、kernel 复杂度上的取舍",
            "设计一套长上下文 needle 评测协议验证量化退化",
        ],
        [
            "为什么 K 和 V 的量化敏感性不同",
            "FP8 为什么常是工程默认首选而不是最低 bit",
            "TurboQuant 的旋转和 inner-product bias 目标解决什么问题",
        ],
    ),
    (
        "sglang",
        "SGLang serving/runtime",
        [
            "画出一次请求从 frontend 到 scheduler 到 KV cache 的路径",
            "对比 RadixAttention 命中和不命中的 token 复用收益",
            "列出一个生产部署需要监控的 8 个指标",
        ],
        [
            "SGLang 的 RadixAttention 解决什么服务端瓶颈",
            "prefill/decode 分离为什么能提升吞吐和隔离尾延迟",
            "如何解释 structured output 对调度和缓存的影响",
        ],
    ),
    (
        "vllm",
        "vLLM serving/runtime",
        [
            "手算 PagedAttention block table 在 3 个序列下的分配",
            "比较 continuous batching 和静态 batching 的吞吐/延迟",
            "设计一次 prefix cache 命中率实验",
        ],
        [
            "PagedAttention 为什么能降低 KV cache 碎片",
            "vLLM V1 的调度和 KV 管理重点是什么",
            "chunked prefill 对 TTFT/TPOT 的影响怎么解释",
        ],
    ),
    (
        "flashattention",
        "FlashAttention IO-aware attention",
        [
            "手推一个 Q block 扫两个 K/V tile 的 online softmax 更新",
            "画出 S tile、P tile、O tile 的生命周期",
            "比较 v1/v2/v3/v4 在 work partition 和 pipeline 上的变化",
        ],
        [
            "FlashAttention 为什么降低 HBM traffic",
            "online softmax 的 m/l/O 三个状态如何保证数值正确",
            "v2 的 work partitioning 相比 v1 改善了什么",
        ],
    ),
    (
        "mlir",
        "MLIR compiler stack",
        [
            "写一个 toy op 的 verifier 和 canonicalization",
            "用 FileCheck 验证一个 lowering pass 的前后 IR",
            "画出 source dialect 到 linalg/gpu/llvm 的合法化边界",
        ],
        [
            "Dialect、Operation、Region、Block、Value 如何组织 IR",
            "PatternRewrite 和 DialectConversion 的区别是什么",
            "为什么 lowering 要分阶段保留语义",
        ],
    ),
    (
        "cutlass",
        "CUTLASS/CuTe GEMM",
        [
            "把 CTA/warp/MMA tile 三层形状标在同一张图上",
            "对比 threadblock swizzle 对 L2 locality 的影响",
            "解释 epilogue fusion 如何减少 global memory 写读",
        ],
        [
            "CUTLASS 的层级 tiling 如何映射到 tensor core",
            "CuTe layout 解决了什么模板表达问题",
            "mainloop 和 epilogue 分别负责什么",
        ],
    ),
    (
        "pytorch",
        "PyTorch compile stack",
        [
            "找一个 torch.compile graph break 并解释原因",
            "画出 Dynamo -> FX -> AOTAutograd -> Inductor -> Triton",
            "比较 eager 与 compile 在一次 elementwise fusion 中的 kernel 数",
        ],
        [
            "Dynamo guard 是什么, 失效时会发生什么",
            "FX graph 在编译链路里扮演什么角色",
            "Inductor 为什么常生成 Triton kernel",
        ],
    ),
    (
        "nccl",
        "NCCL collective communication",
        [
            "画出 ring all-reduce 在 4 卡上的 reduce-scatter/all-gather",
            "比较 ring 和 tree 在带宽/延迟上的适用场景",
            "列出一次 NCCL hang 的排查 checklist",
        ],
        [
            "AllReduce 为什么可以拆成 reduce-scatter 和 all-gather",
            "拓扑感知对多机多卡通信有什么影响",
            "如何区分网络瓶颈和 GPU kernel 瓶颈",
        ],
    ),
    (
        "ldmatrix",
        "ldmatrix / tensor core layout",
        [
            "画出 8x8 tile 从 shared memory 到 register 的 lane 分布",
            "解释 transpose variant 对矩阵布局的影响",
            "对照 mma 指令需要的 fragment layout 做一次手算",
        ],
        [
            "ldmatrix 解决了 tensor core 前的数据装载什么问题",
            "shared memory bank conflict 如何影响 ldmatrix",
            "TV layout 和 MMA fragment layout 如何对应",
        ],
    ),
    (
        "ai_infra",
        "AI Infra 学习路线",
        [
            "把一个 LLM serving 请求拆成调度、KV、通信、kernel 四层",
            "给自己的简历项目标注 compiler/runtime/kernel/cluster 边界",
            "为一个新论文写 5 个可验证实验问题",
        ],
        [
            "AI Infra 面试怎样从系统路径讲到 kernel 细节",
            "吞吐、延迟、显存、质量四个目标如何权衡",
            "如何把论文结论转成工程验证计划",
        ],
    ),
]


DEFAULT_TOPIC = (
    "GPU/AI Infra 主题",
    [
        "用自己的话重画本图的数据流或控制流",
        "补一个最小可验证实验, 明确输入、输出和指标",
        "列出 3 个容易误解的边界条件并给出反例",
    ],
    [
        "这个主题解决的核心瓶颈是什么",
        "关键抽象如何映射到硬件或系统组件",
        "如果线上结果不符合预期, 你会先看哪些证据",
    ],
)


def element_base(element_id: str, element_type: str, x: float, y: float, w: float, h: float, **style: object) -> dict:
    base = {
        "id": element_id,
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
        "seed": style.pop("seed", 91001),
        "version": 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }
    if element_type in {"rectangle", "diamond"}:
        base["roundness"] = {"type": 3}
    base.update(style)
    return base


def text_element(element_id: str, text: str, x: float, y: float, w: float, h: float, size: int = 18, color: str = PALETTE["text"], align: str = "left") -> dict:
    el = element_base(
        element_id,
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
            "verticalAlign": "top",
            "containerId": None,
            "originalText": text,
            "autoResize": False,
            "lineHeight": 1.25,
        }
    )
    return el


def pick_topic(path: Path) -> tuple[str, list[str], list[str]]:
    name = path.as_posix().lower()
    normalized = name.replace("_", "").replace("-", "")
    for token, label, exercises, interviews in TOPIC_RULES:
        if token in normalized:
            return label, exercises, interviews
    return DEFAULT_TOPIC


def bounds(elements: list[dict]) -> tuple[float, float, float]:
    xs = [float(el.get("x", 0)) for el in elements if not el.get("isDeleted")]
    rights = [float(el.get("x", 0)) + abs(float(el.get("width", 0))) for el in elements if not el.get("isDeleted")]
    bottoms = [float(el.get("y", 0)) + abs(float(el.get("height", 0))) for el in elements if not el.get("isDeleted")]
    min_x = min(xs or [80])
    max_x = max(rights or [1600])
    max_y = max(bottoms or [900])
    return min_x, max_x, max_y


def append_panel(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    elements = [el for el in data.get("elements", []) if not str(el.get("id", "")).startswith(PANEL_PREFIX)]
    for el in elements:
        if el.get("type") == "text":
            el["fontFamily"] = 5
            el["fontSize"] = max(14, int(el.get("fontSize") or 18))
            el["lineHeight"] = el.get("lineHeight") or 1.25
    label, exercises, interviews = pick_topic(path)
    min_x, max_x, max_y = bounds(elements)
    x = max(80, min_x)
    width = max(1280, min(1580, max_x - x))
    y = max_y + 80
    panel_height = 300
    stem = "".join(ch if ch.isalnum() else "-" for ch in path.stem.lower())[:48]
    prefix = f"{PANEL_PREFIX}{stem}"
    practice_text = "练习题\n" + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(exercises))
    interview_text = "面试题\n" + "\n".join(f"{i + 1}. {item}" for i, item in enumerate(interviews))
    additions = [
        element_base(f"{prefix}-bg", "rectangle", x, y, width, panel_height, strokeColor="#64748b", backgroundColor=PALETTE["slate"], opacity=70, seed=93001),
        text_element(f"{prefix}-title", f"练习题与面试题补充：{label}", x + 28, y + 22, width - 56, 38, size=24, color=PALETTE["title"]),
        element_base(f"{prefix}-practice-box", "rectangle", x + 28, y + 78, (width - 84) / 2, 190, strokeColor="#2563eb", backgroundColor=PALETTE["blue"], seed=93002),
        text_element(f"{prefix}-practice-text", practice_text, x + 50, y + 98, (width - 130) / 2, 150, size=17, color=PALETTE["text"]),
        element_base(f"{prefix}-interview-box", "rectangle", x + 56 + (width - 84) / 2, y + 78, (width - 84) / 2, 190, strokeColor="#7c3aed", backgroundColor=PALETTE["purple"], seed=93003),
        text_element(f"{prefix}-interview-text", interview_text, x + 78 + (width - 84) / 2, y + 98, (width - 130) / 2, 150, size=17, color=PALETTE["text"]),
    ]
    data["elements"] = elements + additions
    data.setdefault("appState", {}).setdefault("viewBackgroundColor", "#ffffff")
    data.setdefault("files", {})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    files = sorted(ROOT.rglob("*.excalidraw"))
    for path in files:
        append_panel(path)
    print(f"enriched {len(files)} excalidraw files")


if __name__ == "__main__":
    main()
