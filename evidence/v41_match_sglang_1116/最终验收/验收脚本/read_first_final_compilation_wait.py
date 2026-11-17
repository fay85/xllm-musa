from pathlib import Path
import json,datetime
c=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control');out={}
a=datetime.datetime.fromisoformat('2026-11-17T20:55:00+08:00');b=datetime.datetime.fromisoformat('2026-11-17T20:59:40+08:00')
for h in ['sj9','sj10']:
 rows=[]
 for line in (c/'compilation_monitor'/h/'observations.jsonl').read_text().splitlines():
  x=json.loads(line);t=datetime.datetime.fromisoformat(x['at'])
  if a<=t<=b and not x['compilation_clear']:rows.append(x)
 out[h]=rows
print(json.dumps(out))
