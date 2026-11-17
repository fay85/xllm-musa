from pathlib import Path
import json,datetime
r=Path('/expt/runs/p55_draft_gate_on_v1');z={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['formal_client/finished.json','formal_client/summary.json','formal_client/error.txt','trace/finished.json','trace/error.txt']:
 p=r/n
 if p.exists():z[n]=p.read_text()
for host in ['sj9','sj10']:
 p=r/host/'status.json'
 if p.exists():z[host]=json.loads(p.read_text())
print(json.dumps(z))
