# -*- coding: utf-8 -*-
"""
Apply Clean Titles and Core Operation + Workflow Diagrams across all 88 lessons.
Strictly ensures:
1. Frontmatter title has no redundant track name prefix: '第 X 课：<Topic>'
2. Markdown body has no redundant H1 header
3. Every lesson has at least 2 clean, clear illustrations:
   - Fig 1: 核心操作概念图 (Core Operation Concept Diagram, including scientific-figure-making charts)
   - Fig 2: 关键流程图 (Key Execution Workflow Diagram)
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

def update_frontmatter(fm_text, new_title):
    lines = fm_text.strip().split("\n")
    new_lines = []
    title_updated = False
    for line in lines:
        if line.startswith("title:"):
            new_lines.append(f'title: "{new_title}"')
            title_updated = True
        elif line.startswith("subtitle:"):
            # Omit redundant subtitle if it contains track name
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

    content = fpath.read_text(encoding="utf-8")
    
    # 1. Split frontmatter and body
    if not content.startswith("---"):
        raise ValueError(f"No frontmatter in {fpath}")
    
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Malformed frontmatter in {fpath}")
    
    fm = parts[1]
    body = parts[2]
    
    # 2. Update frontmatter
    new_fm = update_frontmatter(fm, lesson_data["title"])
    
    # 3. Clean body H1 headings matching '# 第 X 课：...' at the top of the body
    body_lines = body.split("\n")
    cleaned_body_lines = []
    h1_removed = False
    
    for i, line in enumerate(body_lines):
        if not h1_removed and re.match(r"^#\s+(第\s*\d+\s*课.*|.*主线\s*·\s*第\s*\d+.*)", line.strip()):
            h1_removed = True
            continue # Remove redundant H1
        cleaned_body_lines.append(line)
        
    body = "\n".join(cleaned_body_lines)
    
    # 4. Insert or update diagrams
    fig1 = lesson_data["fig1"].strip()
    fig2 = lesson_data["fig2"].strip()
    diagram_block = f"\n### 核心操作与执行流程图解\n\n{fig1}\n\n{fig2}\n"
    
    if track == "cutlass":
        # In cutlass, replace '## 图解...' section before '## 具体演示'
        if "## 图解" in body:
            pattern = r"## 图解[\s\S]*?(?=## 具体演示)"
            replacement = f"## 核心操作与流程图解\n\n{fig1}\n\n{fig2}\n\n"
            body = re.sub(pattern, replacement, body, count=1)
        else:
            # Fallback insert under 核心心智模型
            if "## 核心心智模型" in body:
                body = body.replace("## 核心心智模型", f"## 核心心智模型\n\n{fig1}\n\n{fig2}\n")
    elif track == "train":
        # In train, check lesson01/02/03/05
        # If ## 核心概念 exists
        if "## 核心概念" in body:
            # Check if there is an existing img-card
            pattern = r"(## 核心概念\s*\n)(::: \{\.img-card\}[\s\S]*?:::|\`\`\`\{mermaid\}[\s\S]*?\`\`\`)?"
            match = re.search(r"## 核心概念", body)
            if match:
                idx = match.end()
                # If there's already an existing diagram right after ## 核心概念, check if fig2 is needed
                # To be cleanest, insert fig1 and fig2 cleanly right after ## 核心概念
                # If existing body already has fig1 text (like llm_memory_breakdown), replace the existing block
                if "::: {.img-card}" in body[idx:idx+1000] or "mermaid" in body[idx:idx+1000]:
                    # Replace the existing card/mermaid with fig1 + fig2
                    sub_pat = r"## 核心概念\s*\n\s*(::: \{\.img-card\}[\s\S]*?:::\s*|\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`\s*)*"
                    body = re.sub(sub_pat, f"## 核心概念\n\n{fig1}\n\n{fig2}\n\n", body, count=1)
                else:
                    body = body[:idx] + f"\n\n{fig1}\n\n{fig2}\n\n" + body[idx:]
        else:
            # Fallback insert before ## 本课目标
            if "## 本课目标" in body:
                body = body.replace("## 本课目标", f"## 核心操作与流程图解\n\n{fig1}\n\n{fig2}\n\n## 本课目标")
    elif track == "inference":
        if "## 核心对象" in body:
            # Replace existing uncaptioned mermaid if present
            sub_pat = r"## 核心对象\s*\n\s*(\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`\s*)*"
            body = re.sub(sub_pat, f"## 核心对象\n\n{diagram_block}\n\n", body, count=1)
        elif "## 调用链与状态变化" in body:
            body = body.replace("## 调用链与状态变化", f"{diagram_block}\n\n## 调用链与状态变化")
        else:
            if "## 本课目标与通过标准" in body:
                body = body.replace("## 本课目标与通过标准", f"## 核心操作与流程图解\n\n{fig1}\n\n{fig2}\n\n## 本课目标与通过标准")
    else:
        # cuda, triton, mlir, rl, runtime, platform
        if "## 核心心智模型" in body:
            # Check if there is an existing uncaptioned mermaid right after ## 核心心智模型
            sub_pat = r"## 核心心智模型\s*\n\s*(\`\`\`(?:\{mermaid\}|mermaid)[\s\S]*?\`\`\`\s*)*"
            body = re.sub(sub_pat, f"## 核心心智模型\n\n{diagram_block}\n\n", body, count=1)
        elif "## 前置关系" in body:
            body = body.replace("## 前置关系", f"{diagram_block}\n\n## 前置关系")
        else:
            body = f"{diagram_block}\n\n" + body

    new_content = f"---{new_fm}---{body}"
    
    # 5. Verification
    mermaid_cnt = new_content.count("```mermaid") + new_content.count("```{mermaid}")
    img_cnt = new_content.count("<img ") + new_content.count("![")
    total_diag = mermaid_cnt + img_cnt
    
    if not dry_run:
        fpath.write_text(new_content, encoding="utf-8")
        
    return {
        "file": str(fpath),
        "title": lesson_data["title"],
        "h1_removed": h1_removed,
        "total_diagrams": total_diag,
        "mermaid_cnt": mermaid_cnt,
        "img_cnt": img_cnt
    }

def main(dry_run=True):
    print(f"Running apply_all_diagrams (dry_run={dry_run})...")
    total_processed = 0
    min_diagrams = 999
    less_than_two = []
    
    for track, lesson_dict in TRACK_DICTS:
        for fname, ldata in lesson_dict.items():
            res = process_file(track, fname, ldata, dry_run=dry_run)
            total_processed += 1
            tot = res["total_diagrams"]
            if tot < min_diagrams:
                min_diagrams = tot
            if tot < 2:
                less_than_two.append(res)
                
    print(f"Total files processed: {total_processed}")
    print(f"Minimum diagrams in any file: {min_diagrams}")
    if less_than_two:
        print(f"WARNING: {len(less_than_two)} files have < 2 diagrams:")
        for item in less_than_two:
            print(" ", item)
    else:
        print("SUCCESS: Every single lesson has >= 2 clean, clear illustrations!")

if __name__ == "__main__":
    import sys
    dry = "--apply" not in sys.argv
    main(dry_run=dry)
