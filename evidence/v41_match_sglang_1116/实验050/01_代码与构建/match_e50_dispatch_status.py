from pathlib import Path
import json
r=Path('/expt/diagnostics/e50_production_dispatch_gpu_v1');z={}
for name in ['job.json','child.json','finished.json','worker.log','test.log']:
 p=r/name
 if p.exists():z[name]=p.read_text()[-4000:]
p=r/'result.json'
if p.exists():
 s=json.loads(p.read_text());z['result']={k:v for k,v in s.items() if k not in ['checks','graphs']};z['check_count']=len(s['checks']);z['graph_count']=len(s['graphs'])
print(json.dumps(z))
