from pathlib import Path
import json,datetime
r=Path('/expt/builds/e52_rope_table_production_module_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','worker.log','cpp.log','cu.log','link.log','import.log']:
 p=r/n
 if p.exists():v[n]=p.read_text(errors='replace')[-1400:]
print(json.dumps(v))
