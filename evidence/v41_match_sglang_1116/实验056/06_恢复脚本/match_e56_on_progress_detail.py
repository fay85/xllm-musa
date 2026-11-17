from pathlib import Path
import json,datetime
r=Path('/expt/runs/p56_short_indexer_on_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['active.json','finished.json','summary.json','error.txt','formal_warmup_compilation_gate.json','formal_compilation_gate.json']:
 p=r/'formal_client'/n
 if p.exists():v[n]=p.read_text()[-2500:]
for host in ['sj9','sj10']:
 p=r/host/'rank0.log';v[host+'_log_tail']=p.read_text(errors='replace')[-1600:]
print(json.dumps(v,ensure_ascii=False))
