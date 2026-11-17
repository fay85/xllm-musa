from pathlib import Path
import json,datetime
r=Path('/expt/runs/final_long_v1');v={'at':datetime.datetime.now().astimezone().isoformat(),'run':r.name}
for h in ['sj9','sj10']:
 d=r/h;v[h]={}
 for name in ['status.json','result.json']:
  if (d/name).exists():v[h][name]=json.loads((d/name).read_text())
 p=d/'rank0.log'
 if p.exists():v[h]['tail']=p.read_text(errors='replace')[-700:]
for name in ['finished.json','summary.json','error.txt','status.json']:
 p=r/'formal_client'/name
 if p.exists():v[name]=p.read_text()
p=r/'formal_client/basic_gate.json'
if p.exists():v['basic']=[{'answer':x['answer'],'passed':x['passed']} for x in json.loads(p.read_text())]
for p in (r/'formal_client').glob('*_compilation_gate.json'):
 z=json.loads(p.read_text());v[p.name]={'attempts':len(z),'first':z[0] if z else None,'last':z[-1] if z else None}
p=r/'formal_client/active.json'
if p.exists():v['active']=json.loads(p.read_text())
print(json.dumps(v,ensure_ascii=False))
