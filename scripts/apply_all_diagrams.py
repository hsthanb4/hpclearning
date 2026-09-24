# -*- coding: utf-8 -*-
"""
Apply Clean Titles and Core Operation + Workflow Diagrams across all 88 lessons.
Strictly ensures:
1. Frontmatter title has no redundant track name prefix: '第 X 课：<Topic>'
2. Markdown body has no redundant H1 header
3. Every lesson has exactly 1 diagram section ('### 核心操作与执行流程图解') with at least 2 clean illustrations:
   - Fig 1: 核心操作概念图 (Core Operation Concept Diagram, including scientific-figure-making charts)
   - Fig 2: 关键流程图 (Key Execution Workflow Diagram)
4. All Mermaid code blocks use ```{mermaid} (not ```mermaid) for native Quarto SVG rendering
5. Zero duplicate captions or duplicated diagram sections
"""

import os
import sys
import re
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_train import TRAIN_LESSONS
from scripts.data_cuda import CUDA_LESSONS
from scripts.data_cutlass import CUTLASS_LESSONS
from scripts.data_triton import TRITON_LESSONS
from scripts.data_mlir import MLIR_LESSONS
from scripts.data_rl import RL_LESSONS
from scripts.data_inference import INFERENCE_LESSONS
from scripts.data_runtime import RUNTIME_LESSONS
from scripts.data_platform import PLATFORM_LESSONS

TRACK_DICTS = [
    ("train", TRAIN_LESSONS),
    ("cuda", CUDA_LESSONS),
    ("cutlass", CUTLASS_LESSONS),
    ("triton", TRITON_LESSONS),
    ("mlir", MLIR_LESSONS),
    ("rl", RL_LESSONS),
    ("inference", INFERENCE_LESSONS),
    ("runtime", RUNTIME_LESSONS),
    ("platform", PLATFORM_LESSONS),
]

def purge_diagram_artifacts(txt, track, rel_path):
    p_header = r"### 核心操作与执行流程图解\s*\n(?:(?:\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`|::: \{\.img-card\}[\s\S]*?:::|<p class=\"caption\"[^>]*>.*?</p>|\s+)*)"
    txt = re.sub(p_header, "", txt)
    
    txt = re.sub(r"## 核心操作与流程图解\s*\n[\s\S]*?(?=## 具体演示)", "", txt)
    txt = re.sub(r"## 核心心智模型\s*\n(?:(?:\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`|::: \{\.img-card\}[\s\S]*?:::|<p class=\"caption\"[^>]*>.*?</p>|\s+)*)(?=### 1\.)", "## 核心心智模型\n\n", txt)
    
    if rel_path == "train/lesson01.qmd":
        txt = re.sub(r"## 核心概念\s*\n(?:(?:\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`|::: \{\.img-card\}[\s\S]*?:::|<p class=\"caption\"[^>]*>.*?</p>|\s+)*)(?=一次普通训练迭代包括：)", "## 核心概念\n\n", txt)
    else:
        txt = re.sub(r"## 核心概念\s*\n(?:(?:\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`|::: \{\.img-card\}[\s\S]*?:::|<p class=\"caption\"[^>]*>.*?</p>|\s+)*)(?=### 1\.)", "## 核心概念\n\n", txt)
        
    if rel_path == "train/lesson02.qmd":
        txt = re.sub(r"::: \{\.img-card\}\s*\n!\[\]\(assets/activation_recompute\.jpg\)[^\n]*\n<p class=\"caption\">[^<]*</p>\s*\n:::\s*", "", txt)

    return txt

def update_frontmatter(fm_text, new_title):
    lines = fm_text.strip().split("\n")
    new_lines = []
    title_updated = False
    for line in lines:
        if line.startswith("title:"):
            new_lines.append(f'title: "{new_title}"')
            title_updated = True
        elif line.startswith("subtitle:"):
            continue
        else:
            new_lines.append(line)
    if not title_updated:
        new_lines.insert(0, f'title: "{new_title}"')
    return "\n" + "\n".join(new_lines) + "\n"

def process_file(track, fname, lesson_data, dry_run=False):
    fpath = Path(track) / fname
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")

    rel_path = f"{track}/{fname}"
    content = fpath.read_text(encoding="utf-8")
    
    # 1. Split frontmatter and body
    if not content.startswith("---"):
        raise ValueError(f"No frontmatter in {fpath}")
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Malformed frontmatter in {fpath}")
    
    fm, body = parts[1], parts[2]
    new_fm = update_frontmatter(fm, lesson_data["title"])
    
    # 2. Purge old diagrams and duplicate captions from body
    cleaned_body = purge_diagram_artifacts(body, track, rel_path)
    
    # 3. Clean body redundant H1 headings
    cleaned_body = re.sub(r"\n#\s+(第\s*\d+\s*课.*|.*主线\s*·\s*第\s*\d+.*)\n", "\n", cleaned_body)
    
    # 4. Insert diagrams
    fig1 = lesson_data["fig1"].strip()
    fig2 = lesson_data["fig2"].strip()
    diagram_block = f"\n### 核心操作与执行流程图解\n\n{fig1}\n\n{fig2}\n"
    
    if track == "train":
        cleaned_body = cleaned_body.replace("## 核心概念\n", f"## 核心概念\n\n{diagram_block}\n", 1)
    elif track in ["cuda", "cutlass", "triton", "mlir", "rl", "runtime", "platform"]:
        cleaned_body = cleaned_body.replace("## 核心心智模型\n", f"## 核心心智模型\n\n{diagram_block}\n", 1)
    elif track == "inference":
        cleaned_body = cleaned_body.replace("## 核心对象\n", f"## 核心对象\n\n{diagram_block}\n", 1)

    # 5. Ensure collapsible answer section
    if "## 参考答案（仅 answer 分支）" in cleaned_body and "::: {.callout-tip" not in cleaned_body:
        parts_ans = cleaned_body.split("## 参考答案（仅 answer 分支）")
        ans_content = parts_ans[1].strip()
        cleaned_body = parts_ans[0] + "::: {.callout-tip collapse=\"true\" title=\"💡 点击展开参考答案与详细代码实现\" icon=\"false\"}\n## 参考答案（仅 answer 分支）\n\n" + ans_content + "\n:::\n"

    new_content = f"---{new_fm}---{cleaned_body}"
    
    # 6. Verification
    diag_cnt = new_content.count("### 核心操作与执行流程图解")
    caps_cnt = len(re.findall(r"<p class=\"caption\"", new_content))
    raw_merm = new_content.count("```mermaid\n")
    if diag_cnt != 1 or caps_cnt != 2 or raw_merm > 0:
        raise ValueError(f"Validation failed for {rel_path}: diag_cnt={diag_cnt}, caps_cnt={caps_cnt}, raw_merm={raw_merm}")
        
    if not dry_run:
        fpath.write_text(new_content, encoding="utf-8")
        
    return {
        "file": str(fpath),
        "title": lesson_data["title"],
        "diag_cnt": diag_cnt,
        "caps_cnt": caps_cnt,
    }

def main(dry_run=True):
    mode = "DRY RUN" if dry_run else "APPLYING"
    print(f"[{mode}] Applying diagrams across all 88 lessons...")
    total_processed = 0
    
    for track, lesson_dict in TRACK_DICTS:
        for fname, ldata in lesson_dict.items():
            process_file(track, fname, ldata, dry_run=dry_run)
            total_processed += 1
                
    print(f"SUCCESS: All {total_processed} lessons processed and verified! Each has exactly 1 diagram section, 2 captions, and zero raw mermaid fences.")

if __name__ == "__main__":
    dry = "--apply" not in sys.argv
    main(dry_run=dry)
