from pathlib import Path
import json,datetime
r=Path('/expt/builds/e40_prequant_aot_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['job.json','child.json','finished.json']:
 p=r/n
 if p.exists():
  z=json.loads(p.read_text());v[n]=z
  if 'pid' in z:
   d=Path('/proc')/str(z['pid']);v[n+'_live']=d.exists() and d.joinpath('stat').read_text().split()[21]==z['start_ticks']
p=r/'build.log'
if p.exists():v['log_tail']=p.read_text(errors='replace')[-3000:]
v['completed_shapes']=len(list(Path('/expt/artifacts/e40_prequant_aot_v1').glob('*/manifest.json')))
print(json.dumps(v))
