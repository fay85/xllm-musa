from pathlib import Path
import json,datetime
r=Path('/expt/builds/e56_short_context_module_v1');v={'at':datetime.datetime.now().astimezone().isoformat()}
for n in ['finished.json','worker.log','cpp.log','link.log','import.log']:
 f=r/n
 if f.exists():v[n]=f.read_text(errors='replace')[-2200:]
print(json.dumps(v))
