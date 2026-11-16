from pathlib import Path
import json,datetime
r=Path('/expt/runs/p39_static_activation_on_v1');j=json.loads((r/'formal_client_launch.json').read_text());p=Path('/proc')/str(j['pid']);v={'at':datetime.datetime.now().astimezone().isoformat(),'client_alive_same_start':p.exists() and p.joinpath('stat').read_text().split()[21]==j['start_ticks']}
for h in ['sj9','sj10']:
 log=(r/h/'rank0.log').read_text(errors='replace');v[h+'_tail']=log[-2300:];v[h+'_memory_lines']=[x for x in log.splitlines() if any(k in x.lower() for k in ['available memory','memory available','gpu memory','free memory','allocated','cache capacity','graph captured'])][-15:]
for f in ['formal_client/active.json','formal_client/summary.json','formal_client/finished.json','formal_client/error.txt']:
 p=r/f
 if p.exists():v[f]=p.read_text()[-2500:]
print(json.dumps(v,ensure_ascii=False))
