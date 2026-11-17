from pathlib import Path
import json,datetime
r=Path('/expt/diagnostics/e52_rope_table_gpu_v2');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','child.json','worker.log','test.log']:
 p=r/n
 if p.exists():v[n]=p.read_text(errors='replace')[-3000:]
p=r/'result.json'
if p.exists():
 d=json.loads(p.read_text());v['result']={'status':d['status'],'checks':len(d['checks']),'graphs':len(d['graphs']),'timings':len(d['timings']),'differences':d['differences'][:3],'last_timings':d['timings'][-2:]}
print(json.dumps(v))
