from pathlib import Path
import json,datetime
r=Path('/expt/diagnostics/e40_prequant_graph_refresh_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['job.json','child.json','finished.json']:
 p=r/n
 if p.exists():
  z=json.loads(p.read_text());v[n]=z
  if 'pid' in z:
   q=Path('/proc')/str(z['pid']);v[n+'_live']=q.exists() and q.joinpath('stat').read_text().split()[21]==z['start_ticks']
p=r/'result.json'
if p.exists():
 z=json.loads(p.read_text());v['result']={'status':z['status'],'checks':len(z['checks']),'max_RMS':max(x['rms'] for x in z['checks'])}
p=r/'probe.log'
if p.exists():v['log_tail']=p.read_text()[-2000:]
print(json.dumps(v))
