#!/usr/bin/env python3
import re
import subprocess
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LOCAL_MERMAID = Path("_site/site_libs/quarto-diagram/mermaid.min.js").resolve()

def test_diagram(item):
    qmd, idx, code = item
    html = f"""<!DOCTYPE html><html><head>
<script src="file://{LOCAL_MERMAID}"></script>
</head><body>
<pre class="mermaid">{code}</pre>
<script>mermaid.initialize({{startOnLoad: true}});</script>
</body></html>"""
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        f.write(html)
        tpath = f.name
    cmd = [CHROME, '--headless=new', '--dump-dom', '--virtual-time-budget=2000', f'file://{tpath}']
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        out = res.stdout
    except Exception as e:
        return (str(qmd), idx, code, f"Timeout/Exception: {e}")
        
    if 'Syntax error in text' in out or 'class="error-icon"' in out:
        return (str(qmd), idx, code, "Syntax error in text")
    return None

def main():
    items = []
    for qmd in sorted(Path('.').glob('*/*.qmd')):
        text = qmd.read_text(encoding='utf-8')
        blocks = re.findall(r'```{mermaid}\n(.*?)```', text, re.DOTALL)
        for idx, b in enumerate(blocks):
            items.append((qmd, idx+1, b.strip()))

    print(f"Testing {len(items)} Mermaid diagrams with 4 workers using local mermaid.min.js...")
    failed = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(test_diagram, item): item for item in items}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                failed.append(res)
                print(f"[FAIL] {res[0]} Block #{res[1]}: {res[3]}")

    print(f"\n==========================================")
    print(f"TOTAL FAILED DIAGRAMS: {len(failed)} / {len(items)}")
    print(f"==========================================")
    for f_path, f_idx, code, err in failed:
        print(f"\n--- {f_path} Block #{f_idx} ({err}) ---")
        print(code)

if __name__ == "__main__":
    main()
