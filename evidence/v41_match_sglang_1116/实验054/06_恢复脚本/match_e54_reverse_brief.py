from pathlib import Path
import json,datetime
r=Path('/expt/runs/p54_rope_proposal_reverse_off_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for h in ['sj9','sj10']:
 d=r/h;z={}
 for n in ['status.json','result.json']:
  if (d/n).exists():z[n]=json.loads((d/n).read_text())
 z['log_tail']=(d/'rank0.log').read_text(errors='replace')[-1400:]
 v[h]=z
for n in ['formal_client/finished.json','formal_client/summary.json','formal_client/error.txt','trace/finished.json','trace/error.txt']:
 if (r/n).exists():v[n]=(r/n).read_text()
p=r/'formal_client/basic_gate.json'
if p.exists():v['basic']=[{'answer':x['answer'],'passed':x['passed']} for x in json.loads(p.read_text())]
p=r/'formal_client/status.json'
if p.exists():v['phases']=[{k:x.get(k) for k in ['phase','exit_code','rows']} for x in json.loads(p.read_text())]
p=r/'formal_client/job.json'
if p.exists():
 j=json.loads(p.read_text());proc=Path('/proc',str(j['pid']));v['client_job']=j;v['client_live']=proc.exists() and (proc/'stat').read_text().split()[21]==j['start_ticks']
print(json.dumps(v,ensure_ascii=False))
