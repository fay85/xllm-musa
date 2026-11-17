from pathlib import Path
import json,datetime
r=Path('/expt/runs/p56_short_indexer_off_v1');z={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['formal_client/finished.json','formal_client/summary.json','formal_client/error.txt','trace/finished.json','trace/error.txt']:
 p=r/n
 if p.exists():z[n]=p.read_text()
for host in ['sj9','sj10']:
 p=r/host/'result.json'
 if not p.exists():p=r/host/'status.json'
 if p.exists():z[host]=json.loads(p.read_text())
p=r/'formal_client/status.json'
if p.exists():z['phases']=[{k:x.get(k) for k in ['phase','exit_code','started_at','finished_at']} for x in json.loads(p.read_text())]
p=r/'formal_client/basic_gate.json'
if p.exists():z['basic']=[{'answer':x['answer'],'passed':x['passed']} for x in json.loads(p.read_text())]
print(json.dumps(z))
