from pathlib import Path
import json,datetime
r=Path('/expt/builds/e55_native_draft_gate_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['launch.json','job.json','finished.json']:
 if (r/n).exists():v[n]=json.loads((r/n).read_text())
if (r/'build.log').exists():v['build_tail']=(r/'build.log').read_text(errors='replace')[-2500:]
print(json.dumps(v))
