from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent


PALETTE = {
    "title": "#1e40af",
    "text": "#374151",
    "muted": "#64748b",
    "line": "#3b82f6",
    "blue": "#a5d8ff",
    "green": "#b2f2bb",
    "orange": "#ffd8a8",
    "purple": "#d0bfff",
    "red": "#ffc9c9",
    "yellow": "#fff3bf",
    "cyan": "#c3fae8",
    "pink": "#eebefa",
    "slate": "#e2e8f0",
}


class Diagram:
    def __init__(self, title: str, width: int = 1600, height: int = 1120) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.elements: list[dict] = []
        self._counter = 0

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    def _base(self, element_type: str, x: float, y: float, w: float, h: float, **style: object) -> dict:
        seed = 1000 + self._counter * 37
        base = {
            "id": self._id(element_type),
            "type": element_type,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": style.pop("strokeColor", "#1e1e1e"),
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
        stroke: str = "#1e1e1e",
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

    def diamond(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "transparent",
        stroke: str = "#1e1e1e",
    ) -> dict:
        el = self._base("diamond", x, y, w, h, strokeColor=stroke, backgroundColor=fill)
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
        stroke: str = "#1e1e1e",
        size: int = 18,
        color: str = PALETTE["text"],
    ) -> None:
        self.rect(x, y, w, h, fill, stroke)
        self.text(text, x + 12, y + 10, w - 24, h - 20, size=size, color=color)

    def note(self, text: str, x: float, y: float, w: float, h: float) -> None:
        self.box(text, x, y, w, h, PALETTE["yellow"], "#f59e0b", size=16)

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = PALETTE["line"],
        stroke_style: str = "solid",
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
                "endArrowhead": "arrow",
            }
        )
        self.elements.append(el)
        return el

    def title_block(self, subtitle: str) -> None:
        self.text(self.title, 80, 42, self.width - 160, 42, size=28, color=PALETTE["title"])
        self.text(subtitle, 80, 88, self.width - 160, 30, size=16, color=PALETTE["muted"])
        self.arrow(90, 128, self.width - 90, 128, "#94a3b8", "dashed")

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


def overview() -> Diagram:
    d = Diagram("SGLang 与 vLLM 推理框架教程总览")
    d.title_block("目标：核心机制 + 面试高频题 + 2026-05-25 前沿功能更新")
    d.rect(70, 160, 700, 560, PALETTE["slate"], "#64748b", opacity=25)
    d.rect(830, 160, 700, 560, PALETTE["slate"], "#64748b", opacity=25)
    d.text("SGLang 教程", 100, 182, 640, 38, size=24, color=PALETTE["title"])
    d.text("vLLM 教程", 860, 182, 640, 38, size=24, color=PALETTE["title"])

    s_boxes = [
        ("定位\n结构化生成语言 + 高性能 runtime\n适合 agentic / RAG / 多轮共享前缀", PALETTE["blue"]),
        ("核心机制\nRadixAttention 前缀树\n连续批处理 + cache-aware scheduling\n结构化输出与工具调用", PALETTE["cyan"]),
        ("前沿重点\nSpec V2 overlap / DFLASH / adaptive spec\nPD disaggregation / decode-side radix cache\nCP/EP/MoE/FA3/FA4/Omni/RL", PALETTE["purple"]),
        ("面试抓手\n解释 prefix reuse 为什么能降 TTFT\n区分 prefill compute-bound 与 decode memory-bound\n会从指标定位瓶颈", PALETTE["green"]),
    ]
    for i, (text, fill) in enumerate(s_boxes):
        d.box(text, 115, 250 + i * 112, 610, 88, fill, "#2563eb", size=17)

    v_boxes = [
        ("定位\n生产级 LLM serving engine\nOpenAI API / offline batch / 多硬件生态", PALETTE["blue"]),
        ("核心机制\nPagedAttention KV block table\nV1 unified scheduler\nchunked prefill + prefix cache", PALETTE["cyan"]),
        ("前沿重点\nv0.21 KV offload + HMA\nthinking budget spec decode\nBlackwell TOKENSPEED_MLA\nNIXL / Mooncake / disagg serving", PALETTE["purple"]),
        ("面试抓手\n讲清 logical block 到 physical block 映射\n讲清 scheduler 如何平衡 TTFT 与吞吐\n会调 max tokens / KV dtype / cache hit", PALETTE["green"]),
    ]
    for i, (text, fill) in enumerate(v_boxes):
        d.box(text, 875, 250 + i * 112, 610, 88, fill, "#2563eb", size=17)

    d.box("共同底层问题\n请求调度\nKV cache\nprefill/decode 分离\nspeculative decoding\n并行与通信\n观测指标", 530, 760, 540, 140, PALETTE["orange"], "#f59e0b", size=20)
    d.arrow(420, 690, 590, 760)
    d.arrow(1180, 690, 1010, 760)
    d.box("学习顺序\n1 基础推理链路\n2 KV cache 与调度\n3 高级特性\n4 前沿 release\n5 面试题反推系统设计", 260, 940, 1080, 100, PALETTE["yellow"], "#f59e0b", size=18)
    return d


def sglang_core() -> Diagram:
    d = Diagram("SGLang 核心机制：从结构化程序到 RadixAttention")
    d.title_block("面试主线：SGLang 的价值不是只跑模型，而是让复杂 LLM 程序自动复用 KV cache")
    stages = [
        ("客户端\nOpenAI API\n或 sgl 程序", PALETTE["blue"]),
        ("前端语言\nprompt + gen\n控制流 + 工具", PALETTE["cyan"]),
        ("调度器\nwaiting/running\ncache-aware", PALETTE["purple"]),
        ("RadixCache\n前缀树匹配\nLRU eviction", PALETTE["green"]),
        ("Runtime\nbatch + memory pool\nCUDA graph", PALETTE["orange"]),
        ("Model Runner\nattention backend\nMoE / quant", PALETTE["red"]),
        ("输出\nstream tokens\nstructured JSON", PALETTE["blue"]),
    ]
    x = 70
    for i, (text, fill) in enumerate(stages):
        d.box(text, x + i * 215, 170, 175, 96, fill, "#1d4ed8", size=16)
        if i < len(stages) - 1:
            d.arrow(x + i * 215 + 176, 218, x + (i + 1) * 215 - 8, 218)

    d.text("RadixAttention 前缀树示意", 90, 330, 640, 36, size=22, color=PALETTE["title"])
    d.box("root", 320, 390, 120, 60, PALETTE["slate"], "#475569", size=18)
    d.box("system prompt\nfew-shot examples", 235, 500, 290, 78, PALETTE["cyan"], "#0f766e", size=17)
    d.box("用户 A 新增 token\n只算增量 KV", 90, 635, 260, 78, PALETTE["green"], "#15803d", size=17)
    d.box("用户 B 新增 token\n共享同一 prefix", 420, 635, 260, 78, PALETTE["green"], "#15803d", size=17)
    d.arrow(380, 450, 380, 500)
    d.arrow(310, 578, 220, 635)
    d.arrow(450, 578, 550, 635)
    d.note("关键点\n相同 token prefix 对应同一段 KV\n命中时跳过重复 prefill\n缓存满时倾向淘汰无引用的叶子", 85, 765, 610, 120)

    d.text("Runtime 为什么快", 815, 330, 650, 36, size=22, color=PALETTE["title"])
    runtime = [
        ("连续批处理\n每轮 forward 接纳新请求\n减少 GPU 空转", PALETTE["blue"]),
        ("结构化输出\n约束解码 / FSM\n减少无效 token 路径", PALETTE["yellow"]),
        ("Speculative decoding\nEAGLE / MTP / NGRAM / DFLASH\n用 draft 降低 target 调用次数", PALETTE["purple"]),
        ("并行策略\nTP / PP / DP / EP / CP\nMoE all-to-all 与长上下文拆分", PALETTE["orange"]),
        ("多后端 kernel\nFA3 / FA4 / TRTLLM / FlashInfer\n按 prefill/decode 选择", PALETTE["red"]),
    ]
    for i, (text, fill) in enumerate(runtime):
        d.box(text, 830 + (i % 2) * 335, 390 + (i // 2) * 135, 305, 95, fill, "#334155", size=16)
    d.box("面试一句话\nSGLang = 可编程前端 + cache-aware runtime。\n核心是把复杂多轮程序拆成可复用 prefix，\n再用调度器和 GPU kernel 把命中变成吞吐。", 835, 820, 620, 145, PALETTE["green"], "#15803d", size=19)
    return d


def sglang_frontier() -> Diagram:
    d = Diagram("SGLang 前沿更新：2026 Q2 功能地图")
    d.title_block("资料基于官方文档、GitHub release v0.5.11 与 2026 Q2 roadmap")
    rows = [
        ("基础栈", "CUDA 13 + Torch 2.11\nDocker / wheel 构建矩阵更新\n多硬件：NVIDIA / AMD / TPU / CPU / XPU / NPU", PALETTE["blue"]),
        ("Attention / Kernel", "FA3 community kernels\nFA4 与 TRTLLM MLA 混合 backend\nDSA sparse attention 与 hybrid attention", PALETTE["red"]),
        ("Spec decode", "Spec V2 overlap scheduler\nDFLASH 初始支持与 ROCm\nadaptive speculative_num_steps\npiecewise CUDA graph 兼容", PALETTE["purple"]),
        ("KV / PD disagg", "decode-side radix cache\nMooncake incremental transfer\nNIXL heterogeneous TP KV transfer\nMamba state slice transfer", PALETTE["cyan"]),
        ("Parallel / MoE / LoRA", "CP all-reduce fusion\nDP attention reduce_scatterv\nDeepSeek-V3 / Kimi-K2 LoRA\nFlashInfer CuteDSL FP4 MoE runner", PALETTE["orange"]),
        ("Diffusion / Omni / RL", "SGLang-Diffusion dynamic batching\nDisaggregated diffusion\nOmni audio / multi-stage runtime\nMiles / verl / slime rollout backend", PALETTE["green"]),
        ("Observability", "OpenTelemetry for spec decode / PP\nPrometheus gRPC metrics\nraw KV token gauges\npending token count in load", PALETTE["yellow"]),
    ]
    d.text("最新 feature 不只是新模型，而是围绕大规模 serving 的五条线：KV、调度、并行、kernel、观测", 100, 152, 1400, 36, size=18, color=PALETTE["muted"])
    for i, (head, body, fill) in enumerate(rows):
        y = 215 + i * 112
        d.box(head, 90, y, 220, 82, fill, "#334155", size=20)
        d.box(body, 335, y, 1110, 82, fill, "#334155", size=17)
        d.arrow(310, y + 41, 335, y + 41)
    d.note("面试表达方式\n不要只背 release 名称。回答时按瓶颈归类：\nTTFT 高 -> prefix / chunked prefill / PD\nTPOT 高 -> decode kernel / KV bandwidth / spec accept\nP99 高 -> scheduler / LoRA drain / disagg transfer / observability", 175, 1015, 1250, 82)
    return d


def sglang_interview() -> Diagram:
    d = Diagram("SGLang 面试常考问题速查")
    d.title_block("问法通常围绕：RadixAttention、调度、结构化输出、spec decode、PD disagg、MoE 并行")
    qa = [
        ("RadixAttention 解决什么", "重复 prompt 的 KV 不重算\n用前缀树记录共享 token 序列"),
        ("和 PagedAttention 区别", "SGLang 强调 trie prefix reuse\nvLLM 强调 block table 内存分页"),
        ("prefill 与 decode 瓶颈", "prefill 更 compute-bound\ndecode 更 KV bandwidth / memory-bound"),
        ("continuous batching", "每轮 forward 动态加入和移出请求\n提升 GPU 利用率"),
        ("结构化输出怎么加速", "约束候选 token\nFSM / grammar 降低非法路径"),
        ("spec decode 原理", "draft 先猜多个 token\ntarget 一次验证并接受前缀"),
        ("PD disaggregation", "prefill 节点算 KV\ndecode 节点拉取 KV 继续生成"),
        ("MoE 并行怎么讲", "EP 负责 expert 分布\nall-to-all 和负载均衡决定尾延迟"),
        ("关键线上指标", "TTFT / TPOT / throughput\ncache hit / KV usage / P99"),
        ("常见坑", "tokenizer 不一致导致 cache miss\nbackend 不匹配导致 kernel 退化"),
    ]
    for i, (q, a) in enumerate(qa):
        col = i % 2
        row = i // 2
        x = 90 + col * 720
        y = 165 + row * 165
        d.box("Q\n" + q, x, y, 230, 112, PALETTE["blue"], "#1d4ed8", size=18)
        d.box("A\n" + a, x + 255, y, 400, 112, PALETTE["green"], "#15803d", size=17)
        d.arrow(x + 230, y + 56, x + 255, y + 56)
    d.note("答题模板\n先说系统瓶颈，再说数据结构，再说调度策略，最后落到指标：\n例如 RadixAttention = KV cache prefix tree + cache-aware scheduling + TTFT/cache-hit 观测。", 190, 1010, 1220, 78)
    return d


def vllm_core() -> Diagram:
    d = Diagram("vLLM 核心机制：PagedAttention 与 V1 引擎")
    d.title_block("面试主线：vLLM 用分页式 KV 管理和统一调度把 GPU batch 做大、做满、做稳")
    stages = [
        ("OpenAI API\nOffline LLM", PALETTE["blue"]),
        ("Processor\nchat template\ntokenizer", PALETTE["cyan"]),
        ("EngineCore\nrequest queue\noutput processor", PALETTE["purple"]),
        ("V1 Scheduler\n统一 token budget\nFCFS / priority", PALETTE["green"]),
        ("KV Manager\nBlockPool\nhash APC", PALETTE["orange"]),
        ("Model Runner V2\nCUDA graph\nattention backend", PALETTE["red"]),
        ("Sampler\nstream response\nmetrics", PALETTE["blue"]),
    ]
    x = 70
    for i, (text, fill) in enumerate(stages):
        d.box(text, x + i * 215, 165, 175, 100, fill, "#1d4ed8", size=16)
        if i < len(stages) - 1:
            d.arrow(x + i * 215 + 176, 215, x + (i + 1) * 215 - 8, 215)

    d.text("PagedAttention：logical block 到 physical block", 100, 330, 650, 36, size=22, color=PALETTE["title"])
    d.box("Request A\nlogical blocks\n0  1  2", 95, 395, 230, 95, PALETTE["blue"], "#1d4ed8", size=18)
    d.box("Block table\n0 -> p7\n1 -> p2\n2 -> p9", 375, 395, 230, 95, PALETTE["yellow"], "#b45309", size=18)
    d.box("GPU physical KV\np2 p7 p9 p13\n可非连续分配", 655, 395, 260, 95, PALETTE["green"], "#15803d", size=18)
    d.arrow(325, 442, 375, 442)
    d.arrow(605, 442, 655, 442)
    d.box("Automatic Prefix Caching\nhash(prefix tokens + block tokens) -> KV Block\n共享 block 通过 ref count 管理", 160, 560, 690, 100, PALETTE["cyan"], "#0f766e", size=18)
    d.arrow(505, 490, 505, 560)

    d.text("V1 scheduler 循环", 1015, 330, 430, 36, size=22, color=PALETTE["title"])
    d.box("等待队列\n新请求", 1015, 395, 200, 70, PALETTE["blue"], "#1d4ed8", size=17)
    d.box("分配 token budget\nprompt 与 output 统一", 1260, 395, 240, 70, PALETTE["yellow"], "#b45309", size=17)
    d.box("一次 forward\nchunked prefill + decode", 1260, 520, 240, 70, PALETTE["purple"], "#6d28d9", size=17)
    d.box("释放 / 命中 / 驱逐\nKV block", 1015, 520, 200, 70, PALETTE["green"], "#15803d", size=17)
    d.arrow(1215, 430, 1260, 430)
    d.arrow(1380, 465, 1380, 520)
    d.arrow(1260, 555, 1215, 555)
    d.arrow(1115, 520, 1115, 465)
    d.note("关键点\nV1 把 prompt token 与 output token 放进同一预算模型，\n因此 chunked prefill、prefix cache、spec decode 能在同一调度循环里协作。", 945, 690, 560, 120)
    d.box("面试一句话\nvLLM = PagedAttention 内存管理 + iteration-level scheduling。\n它把 KV cache 从连续大块变成可共享、可驱逐、可分页的 block。", 210, 825, 1180, 110, PALETTE["orange"], "#f59e0b", size=20)
    return d


def vllm_frontier() -> Diagram:
    d = Diagram("vLLM 前沿更新：v0.21 与 2026 大规模 serving")
    d.title_block("资料基于 GitHub v0.21.0 release、vLLM V1 文档与 2026-04 hybrid SSM disagg 博客")
    rows = [
        ("Breaking / 基础栈", "Transformers v4 deprecated\nC++20 build requirement\nCUDA 13 与 Torch 2.11 生态迁移", PALETTE["red"]),
        ("KV Offload + HMA", "Hybrid Memory Allocator 集成\nscheduler-side sliding window groups\nmulti-connector HMA / MooncakeStoreConnector\nDCP / PCP in OffloadingConnector", PALETTE["cyan"]),
        ("Spec decode", "thinking budget support\nindependent drafter attention backend\nmultimodal support guard\nper-step allocation elimination", PALETTE["purple"]),
        ("Model Runner V2", "Qwen3.5 / Mamba hybrid support\nCUDA graph for ViT / EAGLE prefill\nmetadata rebuild 优化\nRayExecutorV2 default", PALETTE["green"]),
        ("Kernel / Hardware", "Blackwell TOKENSPEED_MLA\nFlashInfer sampler default\nFA4 default MLA prefill from v0.20\nTurboQuant 2-bit KV cache\nNVFP4 KV cache", PALETTE["orange"]),
        ("Disaggregated serving", "bi-directional KV transfer P <-> D\nNIXL transfer redesign and 1.x\nMooncake transfer observability\nhybrid SSM-FA disagg available in v0.20+", PALETTE["blue"]),
        ("API / Frontend", "Responses API streaming tool calling\nstrict tool calling with XGrammar 0.2\nFastokens tokenizer path\nRLHF weight update APIs", PALETTE["yellow"]),
    ]
    d.text("前沿方向：KV offload、disagg、hybrid models、Blackwell kernels、reasoning/spec decode、Responses API", 100, 152, 1400, 36, size=18, color=PALETTE["muted"])
    for i, (head, body, fill) in enumerate(rows):
        y = 215 + i * 112
        d.box(head, 90, y, 245, 82, fill, "#334155", size=19)
        d.box(body, 360, y, 1085, 82, fill, "#334155", size=17)
        d.arrow(335, y + 41, 360, y + 41)
    d.note("面试表达方式\nvLLM 的新 feature 大多围绕两个生产痛点：\n1 KV cache 太贵：offload / quant / APC / disagg\n2 复杂模型变多：Mamba hybrid / multimodal / reasoning / tool calling", 175, 1015, 1250, 82)
    return d


def vllm_interview() -> Diagram:
    d = Diagram("vLLM 面试常考问题速查")
    d.title_block("问法通常围绕：PagedAttention、continuous batching、APC、V1 scheduler、chunked prefill、disagg")
    qa = [
        ("PagedAttention 解决什么", "KV cache 分成固定 block\n逻辑 block 映射到非连续物理 block"),
        ("为什么能提升 batch", "减少碎片与预分配浪费\n同显存容纳更多并发请求"),
        ("APC 怎么做", "hash(prefix + block tokens)\n共享物理 KV block"),
        ("和 RadixAttention 区别", "vLLM 使用 hash block cache\nSGLang 使用前缀树组织共享"),
        ("V1 scheduler 核心", "prompt 与 output 统一 token budget\n支持 FCFS / priority"),
        ("chunked prefill", "长 prompt 分块进入调度\n避免 decode 被长 prefill 饿死"),
        ("spec decode", "drafter 生成候选\ntarget 验证接受前缀"),
        ("PD disaggregation", "prefill worker 产 KV\ndecode worker 通过 NIXL / Mooncake 拉 KV"),
        ("线上怎么调", "max_num_batched_tokens\nmax_num_seqs / gpu_memory_utilization\nkv_cache_dtype / prefix hit"),
        ("如何看瓶颈", "TTFT 看 prefill / queue / cache\nTPOT 看 decode kernel / KV bandwidth\nP99 看调度与通信"),
    ]
    for i, (q, a) in enumerate(qa):
        col = i % 2
        row = i // 2
        x = 90 + col * 720
        y = 165 + row * 165
        d.box("Q\n" + q, x, y, 230, 112, PALETTE["blue"], "#1d4ed8", size=18)
        d.box("A\n" + a, x + 255, y, 400, 112, PALETTE["green"], "#15803d", size=17)
        d.arrow(x + 230, y + 56, x + 255, y + 56)
    d.note("答题模板\n先画 block table，再讲 scheduler loop，最后落到指标和参数。\n这比只说 PagedAttention 是虚拟内存类比更像工程答案。", 190, 1010, 1220, 78)
    return d


def write_readme() -> None:
    text = """# SGLang 与 vLLM Excalidraw 教程

更新时间：2026-05-25

本目录包含 6 张标准 `.excalidraw` 图：

1. `00_overview_sglang_vllm_2026.excalidraw`：学习路线总览
2. `01_sglang_core_mechanisms.excalidraw`：SGLang 核心机制
3. `02_sglang_frontier_features_2026q2.excalidraw`：SGLang 前沿 feature 更新
4. `03_sglang_interview_questions.excalidraw`：SGLang 面试常考题
5. `04_vllm_core_mechanisms.excalidraw`：vLLM 核心机制
6. `05_vllm_frontier_features_2026q2.excalidraw`：vLLM 前沿 feature 更新
7. `06_vllm_interview_questions.excalidraw`：vLLM 面试常考题

主要资料源：

- SGLang docs: https://docs.sglang.io/
- SGLang release notes: https://github.com/sgl-project/sglang/releases
- SGLang Q2 roadmap: https://github.com/sgl-project/sglang/issues/22949
- NVIDIA SGLang overview: https://docs.nvidia.com/deeplearning/frameworks/sglang-release-notes/overview.html
- vLLM docs: https://docs.vllm.ai/
- vLLM release notes: https://github.com/vllm-project/vllm/releases
- vLLM V1 guide: https://docs.vllm.ai/en/stable/usage/v1_guide/
- vLLM hybrid SSM disaggregation blog: https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg

## 练习题

1. 画出一次请求从 HTTP/gRPC 入口到 scheduler、KV cache、decode worker、返回 token 的完整路径，并标出 TTFT、TPOT、queueing latency 分别在哪里产生。
2. 对 SGLang：设计一个 RadixAttention prefix cache 命中实验，分别构造完全命中、部分命中和完全未命中的 prompt 集合。
3. 对 vLLM：手算 3 个不同长度序列在 PagedAttention block table 中的分配和释放过程，说明它如何降低 KV cache 碎片。
4. 对比 prefill/decode disaggregation、chunked prefill、continuous batching 三个机制：分别写出它们优化的瓶颈和可能带来的副作用。

## 面试题

1. SGLang 的 RadixAttention 与普通 prefix cache 的核心差异是什么，为什么适合多轮和结构化请求？
2. vLLM 的 PagedAttention 为什么能提升显存利用率，它和 OS virtual memory 的类比在哪里成立、在哪里不成立？
3. 服务端 LLM runtime 中，scheduler 需要同时平衡哪些目标：吞吐、尾延迟、公平性、显存和 cache 命中率如何取舍？
4. 如果线上 decode TPOT 突然恶化，你会按哪些证据顺序排查：队列、KV cache、GPU kernel、通信、batch shape 还是模型配置？
"""
    (OUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    diagrams = [
        ("00_overview_sglang_vllm_2026.excalidraw", overview()),
        ("01_sglang_core_mechanisms.excalidraw", sglang_core()),
        ("02_sglang_frontier_features_2026q2.excalidraw", sglang_frontier()),
        ("03_sglang_interview_questions.excalidraw", sglang_interview()),
        ("04_vllm_core_mechanisms.excalidraw", vllm_core()),
        ("05_vllm_frontier_features_2026q2.excalidraw", vllm_frontier()),
        ("06_vllm_interview_questions.excalidraw", vllm_interview()),
    ]
    for filename, diagram in diagrams:
        diagram.save(filename)
    write_readme()


if __name__ == "__main__":
    main()
