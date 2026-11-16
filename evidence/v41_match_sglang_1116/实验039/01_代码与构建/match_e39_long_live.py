from pathlib import Path
import json,datetime
r=Path('/expt/runs/p39_static_activation_on_v1');o=r/'long_smoke';j=json.loads((o/'job.json').read_text());p=Path('/proc')/str(j['pid']);v={'at':datetime.datetime.now().astimezone().isoformat(),'same_live':p.exists() and p.joinpath('stat').read_text().split()[21]==j['start_ticks']}
for name in ['finished.json','result.json','error.txt']:
 f=o/name
 if f.exists():v[name]=f.read_text()[:4000]
for h in ['sj9','sj10']:v[h]=(r/h/'rank0.log').read_text(errors='replace')[-1600:]
print(json.dumps(v))
