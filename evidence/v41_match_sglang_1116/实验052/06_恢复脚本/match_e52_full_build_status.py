from pathlib import Path
import json,datetime
r=Path('/expt/builds/e52_native_rope_table_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','job.json','launch.json']:
 p=r/n
 if p.exists():v[n]=json.loads(p.read_text())
p=r/'build.log'
if p.exists():v['tail']=p.read_text(errors='replace')[-2200:]
print(json.dumps(v))
