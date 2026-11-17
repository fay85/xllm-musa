from pathlib import Path
import json,datetime
r=Path('/expt/diagnostics/e52_rope_arithmetic_gpu_v3');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','child.json','worker.log','test.log']:
 p=r/n
 if p.exists():v[n]=p.read_text(errors='replace')[-2500:]
p=r/'result.json'
if p.exists():
 d=json.loads(p.read_text());v['result']={k:d[k] for k in ['status','mismatch_totals','exact_modes']};v['cases']=len(d['cases']);v['last_case']=d['cases'][-1]
print(json.dumps(v))
