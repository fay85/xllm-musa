from pathlib import Path
import json
r=Path('/expt/builds/e51_gather_module_v4_bits');v={}
for n in ['job.json','finished.json','import.log','cpp.log','link.log','worker.log']:
 p=r/n
 if p.exists():v[n]=p.read_text()[-4500:]
print(json.dumps(v))
