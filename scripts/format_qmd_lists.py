# -*- coding: utf-8 -*-
"""
Intelligent Markdown Typographic & List Indentation Formatter for Quarto Books.
Fixes Pandoc list-collapsing bugs by ensuring proper blank lines and indentation:
1. Blank line before lists following paragraphs.
2. Blank line between numbered list items that contain sub-items.
3. Blank line around display math blocks ($$).
4. Clean 3-space indentation for sub-bullets under numbered list items.
5. Strictly preserves code blocks, frontmatter, and tables.
"""

import re
from pathlib import Path

def format_qmd_text(text: str) -> str:
    lines = text.split("\n")
    out = []
    in_code = False
    in_frontmatter = False
    code_fence = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle YAML frontmatter
        if i == 0 and stripped == "---":
            in_frontmatter = True
            out.append(line)
            i += 1
            continue
        if in_frontmatter:
            out.append(line)
            if stripped == "---":
                in_frontmatter = False
            i += 1
            continue

        # Handle Code Fences
        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_fence = stripped[:3]
            elif stripped.startswith(code_fence):
                in_code = False
                code_fence = ""
            out.append(line)
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        # If outside code blocks:
        prev_out = out[-1].strip() if out else ""

        # Rule 1: Display math $$ should have blank lines around it if not inside a list
        if stripped == "$$":
            if prev_out and not prev_out.startswith(("#", ":::", "$$")):
                out.append("")
            out.append(line)
            i += 1
            continue

        # Rule 2: Top-level numbered list (e.g. "1. **...")
        m_num = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m_num:
            # If previous line is non-empty and not already a blank line or heading
            if prev_out and not prev_out.startswith(("#", ":::")):
                out.append("")
            out.append(line)
            i += 1
            continue

        # Rule 3: Top-level bullet list (e.g. "- **..." or "* **...") following a paragraph
        m_bullet = re.match(r"^[-*]\s+(.*)$", line)
        if m_bullet and not line.startswith("---"):
            if prev_out and not prev_out.startswith(("#", ":::", "-", "*", ">", "|")):
                out.append("")
            out.append(line)
            i += 1
            continue

        # Rule 4: Sub-bullets under numbered lists (e.g. "   - ...")
        m_sub = re.match(r"^(\s+)([-*])\s+(.*)$", line)
        if m_sub:
            indent = m_sub.group(1)
            marker = m_sub.group(2)
            content = m_sub.group(3)
            # Normalize to 3 spaces if indentation is 1-4 spaces
            if len(indent) <= 4:
                norm_indent = "   "
            else:
                norm_indent = "      "
            out.append(f"{norm_indent}{marker} {content}")
            i += 1
            continue

        # Default: normal line
        out.append(line)
        i += 1

    # Post-process to eliminate excessive 3+ consecutive blank lines
    res = []
    blank_count = 0
    for l in out:
        if not l.strip():
            blank_count += 1
            if blank_count <= 1:
                res.append("")
        else:
            blank_count = 0
            res.append(l)

    return "\n".join(res)

def format_all_lessons():
    qmd_files = sorted(Path(".").glob("*/*.qmd"))
    modified = []
    
    for f in qmd_files:
        if "_site" in str(f) or "reports" in str(f):
            continue
        orig = f.read_text(encoding="utf-8")
        formatted = format_qmd_text(orig)
        if formatted != orig:
            f.write_text(formatted, encoding="utf-8")
            modified.append(str(f))
            print(f"Formatted: {f}")
            
    # Also root index.qmd if exists
    root_idx = Path("index.qmd")
    if root_idx.exists():
        orig = root_idx.read_text(encoding="utf-8")
        formatted = format_qmd_text(orig)
        if formatted != orig:
            root_idx.write_text(formatted, encoding="utf-8")
            modified.append(str(root_idx))
            print(f"Formatted: {root_idx}")

    print(f"\nTotal files formatted: {len(modified)}")
    return modified

if __name__ == "__main__":
    format_all_lessons()
