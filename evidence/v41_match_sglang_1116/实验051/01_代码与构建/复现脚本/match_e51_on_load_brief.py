from pathlib import Path
import json,datetime
p=Path('/expt/runs/p51_bit_gather_on_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for h in ['sj9','sj10']:
 f=p/h/'rank0.log';text=f.read_text(errors='replace') if f.exists() else '';v[h]=text[-2200:]
for n in ['formal_client/active.json','formal_client/basic_gate.json','formal_client/summary.json','formal_client/finished.json','formal_client/error.txt']:
 f=p/n
 if f.exists():v[n]=f.read_text()[-2500:]
print(json.dumps(v))
