from pathlib import Path
import json,datetime
r=Path('/expt/diagnostics/e53_rope_qdq_production_gpu_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','child.json','worker.log','test.log']:
 p=r/n
 if p.exists():v[n]=p.read_text(errors='replace')[-3000:]
p=r/'result.json'
if p.exists():
 d=json.loads(p.read_text());v['result']={'status':d['status'],'checks':len(d['checks']),'graphs':len(d['graphs']),'timings':len(d['timings']),'projection_checks':len(d.get('projection_checks',[])),'projection_graphs':len(d.get('projection_graphs',[])),'differences':d['differences'][:3],'last_timings':d['timings'][-2:]}
print(json.dumps(v))
