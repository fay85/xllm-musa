from pathlib import Path
import json,datetime
r=Path('/expt/runs/p39_static_activation_on_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for sub in ['formal_client','trace','long_smoke','post_long_checks']:
 for n in ['active.json','summary.json','finished.json','error.txt']:
  p=r/sub/n
  if p.exists():
   z=p.read_text();v[sub+'/'+n]=json.loads(z) if n.endswith('.json') else z
   if n=='active.json':v[sub+'/'+n]={k:x for k,x in v[sub+'/'+n].items() if k!='command'}
for h in ['sj9','sj10']:
 for n in ['status.json','result.json']:
  p=r/h/n
  if p.exists():v[h+'/'+n]=json.loads(p.read_text())
print(json.dumps(v,ensure_ascii=False))
