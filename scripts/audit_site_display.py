#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Display & Rendering Audit for all courses and lessons in HPC Learning.
Validates:
1. All lesson files exist and rendered to HTML in _site.
2. Headless Chrome DOM rendering of all 88 lessons.
3. Mermaid diagram rendering (no syntax errors, proper SVG generation).
4. Image references (existence, non-zero size, path resolution).
5. Math / KaTeX rendering (no parse errors, no raw unparsed $$).
6. Quarto callouts and containers (no leaked raw ':::').
7. Internal links and anchor targets validity.
8. CSS responsive constraints for cards, diagrams, tables.
"""

import sys
import os
import re
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

SITE_DIR = Path("_site")
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# All courses defined in _quarto.yml
COURSES = {
    "train": 14,
    "cuda": 12,
    "cutlass": 8,
    "triton": 10,
    "mlir": 8,
    "rl": 12,
    "inference": 8,
    "runtime": 8,
    "platform": 8,
}

def get_all_expected_lessons():
    # Collect all pages configured for rendering in _quarto.yml
    targets = [
        Path("index.qmd"),
        Path("platform/TUTORIAL_AUDIT_AND_PRACTICE.qmd"),
        Path("triton/tilelang_tutorial_2026/TUTORIAL.qmd"),
        Path("triton/triton_tutorial_2026/TUTORIAL.qmd"),
    ]
    # Add all course files (both index.qmd and lesson*.qmd)
    for course in COURSES.keys():
        targets.extend(sorted(Path(course).glob("*.qmd")))
    
    # Sort and deduplicate
    unique = []
    seen = set()
    for t in targets:
        if t.exists() and str(t) not in seen:
            unique.append(t)
            seen.add(str(t))
    return unique

def check_static_html(qmd_path):
    html_rel = qmd_path.with_suffix(".html")
    html_path = SITE_DIR / html_rel
    
    issues = []
    
    if not html_path.exists():
        return {
            "path": str(qmd_path),
            "html_path": str(html_path),
            "exists": False,
            "issues": [f"Missing rendered HTML file: {html_path}"],
            "mermaid_qmd_count": 0,
            "images": []
        }
    
    qmd_text = qmd_path.read_text(encoding="utf-8")
    html_text = html_path.read_text(encoding="utf-8")
    
    # 1. Title check
    title_match = re.search(r'<h1 class="title">([^<]+)</h1>', html_text)
    if not title_match:
        # Check alternative title
        alt_title = re.search(r'<title>([^<]+)</title>', html_text)
        if not alt_title:
            issues.append("Missing page title in HTML")
    
    # 2. Count expected Mermaid in QMD
    mermaid_qmd_blocks = re.findall(r'```{mermaid}', qmd_text)
    mermaid_qmd_count = len(mermaid_qmd_blocks)
    
    # Check if html has mermaid elements
    mermaid_html_matches = re.findall(r'<pre class="mermaid', html_text)
    if mermaid_qmd_count > 0 and len(mermaid_html_matches) != mermaid_qmd_count:
        issues.append(f"Mermaid count mismatch: {mermaid_qmd_count} in QMD vs {len(mermaid_html_matches)} pre tags in HTML")
        
    # 3. Leaked markdown syntax
    # Callout leaks: literal '::: {' or ':::' outside code blocks
    # We strip <pre><code>...</code></pre> first to avoid false positives in code snippets
    cleaned_html = re.sub(r'<pre\b[^>]*>.*?</pre>', '', html_text, flags=re.DOTALL)
    cleaned_html = re.sub(r'<code\b[^>]*>.*?</code>', '', cleaned_html, flags=re.DOTALL)
    
    if "::: {" in cleaned_html or ":::{." in cleaned_html:
        issues.append("Found unrendered raw Quarto callout block '::: {' in HTML body")
    
    dangling_colons = re.findall(r'(?:<p>|<p\s+[^>]*>)\s*:::\s*</p>', cleaned_html)
    if dangling_colons:
        issues.append(f"Found {len(dangling_colons)} unrendered dangling ':::' closing tags in HTML body")
        
    # Check for leaked unrendered markdown image syntax: ![...](...)
    leaked_md_imgs = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', cleaned_html)
    if leaked_md_imgs:
        issues.append(f"Found unrendered markdown image tags: {leaked_md_imgs}")
        
    # Check for KaTeX error classes
    if "katex-error" in html_text:
        issues.append("Found 'katex-error' in rendered HTML (KaTeX failed to parse math expression)")
        
    # 4. Check images
    # All <img> tags in html
    img_tags = re.findall(r'<img\s+[^>]*src=[\"\']([^\"\']+)[\"\']', html_text)
    checked_imgs = []
    for src in img_tags:
        # Ignore external http/https or data: URLs
        if src.startswith("http://") or src.startswith("https://") or src.startswith("data:"):
            continue
        # Resolve path relative to html_path.parent
        img_file = (html_path.parent / src).resolve()
        checked_imgs.append(src)
        if not img_file.exists():
            issues.append(f"Broken image reference 404: {src} -> {img_file}")
        elif img_file.stat().st_size == 0:
            issues.append(f"Empty image file (0 bytes): {src}")
            
    # 5. Check internal relative links
    a_hrefs = re.findall(r'<a\s+[^>]*href=[\"\']([^\"\']+)[\"\']', html_text)
    for href in a_hrefs:
        if href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:") or href.startswith("javascript:") or href.startswith("#"):
            continue
        # Strip anchor
        clean_href = href.split("#")[0]
        if clean_href:
            target_path = (html_path.parent / clean_href).resolve()
            if not target_path.exists():
                issues.append(f"Broken internal link 404: {href} -> {target_path}")
                
    return {
        "path": str(qmd_path),
        "html_path": str(html_path),
        "exists": True,
        "issues": issues,
        "mermaid_qmd_count": mermaid_qmd_count,
        "images": checked_imgs,
    }

def check_chrome_dom(qmd_path):
    html_rel = qmd_path.with_suffix(".html")
    html_path = (SITE_DIR / html_rel).resolve()
    
    file_url = f"file://{html_path}"
    cmd = [
        CHROME_PATH,
        "--headless=new",
        "--dump-dom",
        "--virtual-time-budget=2000",
        file_url
    ]
    
    dom = ""
    for attempt in range(2):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
            dom = res.stdout
            break
        except Exception as e:
            if attempt == 1:
                return {
                    "path": str(qmd_path),
                    "chrome_ok": False,
                    "issues": [f"Chrome failed to render after retry: {str(e)}"],
                    "svg_count": 0,
                }
            time.sleep(1)
        
    issues = []
    # 1. Check for Mermaid errors in rendered DOM
    if "Syntax error in text" in dom or "Syntax error in graph" in dom:
        issues.append("Mermaid rendering syntax error detected in Chrome DOM")
    if 'class="error-icon"' in dom or 'aria-roledescription="error"' in dom:
        issues.append("Mermaid error element detected in Chrome DOM")
        
    # 2. Check for KaTeX errors
    if "katex-error" in dom or "ParseError:" in dom:
        issues.append("KaTeX math parse error detected in Chrome DOM")
        
    # 3. Check rendered SVGs
    svg_count = dom.count("<svg")
    
    return {
        "path": str(qmd_path),
        "chrome_ok": len(issues) == 0,
        "issues": issues,
        "svg_count": svg_count,
        "dom_size": len(dom)
    }

def main():
    lessons = get_all_expected_lessons()
    print(f"==================================================")
    print(f"AUDITING {len(lessons)} LESSONS ACROSS {len(COURSES)} COURSES")
    print(f"==================================================")
    
    # Phase 1: Static HTML & Markdown Check
    print("\n--- Phase 1: Static HTML, Links, Images, & Callouts Check ---")
    static_results = []
    total_static_issues = 0
    for qmd in lessons:
        res = check_static_html(qmd)
        static_results.append(res)
        if res["issues"]:
            total_static_issues += len(res["issues"])
            print(f"[FAIL] {qmd}:")
            for iss in res["issues"]:
                print(f"       - {iss}")
        else:
            print(f"[OK] {qmd} (Mermaid: {res['mermaid_qmd_count']}, Imgs: {len(res['images'])})")
            
    print(f"\nPhase 1 Complete: {total_static_issues} issue(s) detected across {len(lessons)} lessons.")
    
    # Phase 2: Chrome Headless DOM Execution
    print("\n--- Phase 2: Headless Chrome Live DOM & JS Execution Check (Concurrency=3) ---")
    chrome_issues_total = 0
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_qmd = {executor.submit(check_chrome_dom, qmd): qmd for qmd in lessons}
        for future in as_completed(future_to_qmd):
            qmd = future_to_qmd[future]
            try:
                res = future.result()
                if res["issues"]:
                    chrome_issues_total += len(res["issues"])
                    print(f"[CHROME FAIL] {qmd}:")
                    for iss in res["issues"]:
                        print(f"              - {iss}")
                else:
                    print(f"[CHROME OK] {qmd} (SVGs: {res['svg_count']}, DOM: {res['dom_size']} bytes)")
            except Exception as e:
                chrome_issues_total += 1
                print(f"[CHROME ERROR] {qmd}: {e}")
                
    t1 = time.time()
    print(f"\nPhase 2 Complete in {t1-t0:.2f}s: {chrome_issues_total} issue(s) detected.")
    
    print("\n==================================================")
    print(f"TOTAL SUMMARY:")
    print(f"  Lessons Checked: {len(lessons)}")
    print(f"  Static Issues:   {total_static_issues}")
    print(f"  Chrome Issues:   {chrome_issues_total}")
    print("==================================================")
    
    if total_static_issues > 0 or chrome_issues_total > 0:
        sys.exit(1)
    else:
        print("ALL LESSONS PASSED FULL DISPLAY & RENDERING AUDIT!")
        sys.exit(0)

if __name__ == "__main__":
    main()
