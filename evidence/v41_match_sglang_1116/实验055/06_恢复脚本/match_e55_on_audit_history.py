from pathlib import Path
import json,datetime
root=Path('/expt/00_control');out={}
start=datetime.datetime.fromisoformat('2026-11-17T19:34:55+08:00');end=datetime.datetime.fromisoformat('2026-11-17T19:36:20+08:00')
for h in ['sj9','sj10']:
 rows=[json.loads(s) for s in (root/'compilation_monitor'/h/'observations.jsonl').read_text().splitlines() if s]
 rows=[{'at':v['at'],'clear':v['compilation_clear'],'builds':[{'pid':p['pid'],'comm':p['comm'],'containers':p['containers']} for p in v['build_processes']]} for v in rows if start<=datetime.datetime.fromisoformat(v['at'])<=end]
 out[h]=rows
print(json.dumps(out))
