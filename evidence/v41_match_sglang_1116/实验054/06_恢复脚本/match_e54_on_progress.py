from pathlib import Path
import json,datetime
r=Path('/expt/runs/p54_rope_proposal_on_v1');out={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['formal_client/active.json','formal_client/basic_gate.json','formal_client/summary.json','formal_client/finished.json','formal_client/error.txt']:
 p=r/n
 if p.exists():out[n]=p.read_text()[:3500]
for h in ['sj9','sj10']:
 p=r/h/'rank0.log'
 if p.exists():out[h+'_log_tail']=p.read_text(errors='replace')[-1600:]
print(json.dumps(out))

