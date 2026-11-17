from pathlib import Path
import json,datetime
r=Path('/expt/runs/p53_rope_qdq_off_v1');o=r/'trace';v={'at':datetime.datetime.now().astimezone().isoformat()}
for f in ['finished.json','error.txt','requests.json']:
 p=o/f
 if p.exists():v[f]=p.read_text()
p=o/'job.json'
if p.exists():
 j=json.loads(p.read_text());q=Path('/proc')/str(j['pid']);v['live']=q.exists() and q.joinpath('stat').read_text().split()[21]==str(j['start_ticks'])
print(json.dumps(v))
