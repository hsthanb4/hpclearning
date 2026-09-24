import re
from pathlib import Path

print('=== Scanning for potential Mermaid syntax pitfalls in all QMD and data_*.py files ===')

def check_text(source_name, text):
    # Match both qmd blocks and python string literals
    if source_name.endswith('.qmd'):
        blocks = re.findall(r'```{mermaid}(.*?)```', text, re.DOTALL)
    else:
        blocks = re.findall(r'```{mermaid}\n(.*?)```', text, re.DOTALL)
        
    for b_idx, block in enumerate(blocks):
        lines = block.split('\n')
        for l_idx, line in enumerate(lines):
            # Check edge labels with parentheses without quotes
            m_edge = re.findall(r'\|([^|\"]*[\(\)][^|\"]*)\|', line)
            if m_edge:
                print(f'{source_name} [block {b_idx+1} line {l_idx+1}]: unquoted parens in edge: {line.strip()}')
            # Check unicode arrows
            if any(ch in line for ch in ['➔', '➜', '➤', '➡', '←', '→', '↔']):
                print(f'{source_name} [block {b_idx+1} line {l_idx+1}]: unicode arrow: {line.strip()}')
            # Check invalid arrows
            if '<===>' in line or '====' in line:
                print(f'{source_name} [block {b_idx+1} line {l_idx+1}]: invalid arrow: {line.strip()}')
            # Check sequence diagram reserved keywords as participant
            if 'sequenceDiagram' in block and re.search(r'participant\s+Opt\b', line):
                print(f'{source_name} [block {b_idx+1} line {l_idx+1}]: participant Opt reserved keyword: {line.strip()}')

for p in sorted(Path('.').glob('**/*.qmd')):
    if '.git' in p.parts or '_site' in p.parts:
        continue
    check_text(str(p), p.read_text(encoding='utf-8'))

for p in sorted(Path('scripts').glob('data_*.py')):
    check_text(str(p), p.read_text(encoding='utf-8'))
