from pathlib import Path
import json,datetime
r=Path('/expt/runs/p50_gate_simt_on_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for name in ['trace/job.json','trace/finished.json','trace/error.txt','trace/requests.json','trace/start_profile.json','trace/stop_profile.json']:
 p=r/name
 if p.exists():v[name]=p.read_text()[-6000:]
for h in ['sj9','sj10']:v[h]=json.loads((r/h/'status.json').read_text())
print(json.dumps(v))
