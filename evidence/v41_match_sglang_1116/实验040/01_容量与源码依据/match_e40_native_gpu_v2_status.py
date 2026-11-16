from pathlib import Path
import json,datetime
r=Path('/expt/diagnostics/e40_prequant_native_gpu_v2');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['job.json','child.json','finished.json']:
 p=r/n
 if p.exists():
  z=json.loads(p.read_text());v[n]=z
  if 'pid' in z:
   d=Path('/proc')/str(z['pid']);v[n+'_live']=d.exists() and d.joinpath('stat').read_text().split()[21]==z['start_ticks']
p=r/'partial.json'
if p.exists():
 z=json.loads(p.read_text());v['checks']=len(z['checks']);v['benchmarks']=z['benchmarks'][-2:]
p=r/'probe.log'
if p.exists():v['log_tail']=p.read_text(errors='replace')[-4500:]
print(json.dumps(v))
