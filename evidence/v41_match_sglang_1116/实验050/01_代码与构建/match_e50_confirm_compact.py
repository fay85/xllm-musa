from pathlib import Path
import json,datetime
r=Path('/expt/runs/p50_gate_simt_confirm_on_v1');o=r/'formal_client';v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','summary.json','error.txt','active.json']:
 p=o/n
 if p.exists():v[n]=p.read_text()[-3500:]
for h in ['sj9','sj10']:v[h]=json.loads((r/h/'status.json').read_text())
print(json.dumps(v))
