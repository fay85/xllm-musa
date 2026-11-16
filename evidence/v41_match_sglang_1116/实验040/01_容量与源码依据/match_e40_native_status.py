from pathlib import Path
import json,datetime
r=Path('/expt/builds/e40_prequant_native_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['job.json','finished.json']:
 p=r/n
 if p.exists():
  z=json.loads(p.read_text());v[n]=z
  if 'pid' in z:
   d=Path('/proc')/str(z['pid']);v[n+'_live']=d.exists() and d.joinpath('stat').read_text().split()[21]==z['start_ticks']
for p in r.glob('*.log'):v[p.name]=p.read_text(errors='replace')[-1500:]
v['objects']=[p.name for p in r.glob('*.o')];print(json.dumps(v))
