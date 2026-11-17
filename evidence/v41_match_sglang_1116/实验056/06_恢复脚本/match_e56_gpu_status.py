from pathlib import Path
import json,datetime
r=Path('/expt/diagnostics/e56_short_context_gpu_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','status.json','worker.log','child.json']:
 f=r/n
 if f.exists():v[n]=f.read_text()[-2600:]
for f in r.glob('*.log'):
 v[f.name]=f.read_text(errors='replace')[-1600:]
print(json.dumps(v))
