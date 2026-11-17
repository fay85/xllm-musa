from pathlib import Path
import json
root=Path('/expt/00_control/compilation_monitor');out={}
for host in ['sj9','sj10']:
 rows=[json.loads(x) for x in (root/host/'observations.jsonl').read_text().splitlines() if x]
 busy=[v for v in rows if '2026-11-17T20:19:00'<=v['at']<'2026-11-17T20:25:00' and not v['compilation_clear']]
 out[host]={'samples':len(busy),'first':busy[0] if busy else None,'last':busy[-1] if busy else None}
print(json.dumps(out))
