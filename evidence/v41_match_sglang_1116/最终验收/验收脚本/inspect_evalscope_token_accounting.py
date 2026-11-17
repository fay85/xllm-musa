from pathlib import Path
import json
root=Path('/work/artifacts/bench_venv_v2/lib/python3.10/site-packages/evalscope/perf')
files=[]
for p in root.rglob('*.py'):
 text=p.read_text()
 if ('usage.completion_tokens' in text or "usage.get('completion_tokens'" in text or 'usage["completion_tokens"]' in text or 'usage.prompt_tokens' in text):
  lines=text.splitlines();selected=set()
  for i,line in enumerate(lines):
   if 'usage' in line or 'completion_tokens' in line or 'tokenizer' in line:selected.update(range(max(0,i-3),min(len(lines),i+5)))
  files.append({'path':str(p),'lines':[(i+1,lines[i]) for i in sorted(selected)]})
print(json.dumps(files))

