from pathlib import Path
import json
r=Path('/expt/runs/p50_gate_simt_on_v1');v={}
for n in ['long_smoke','runtime_timing/armed','formal_client/finished.json','trace/finished.json','long_smoke/job.json','long_smoke/finished.json','long_smoke/result.json','long_smoke_supervisor.log']:
 p=r/n;v[n]={'exists':p.exists(),'file':p.read_text()[-5000:] if p.is_file() else None}
print(json.dumps(v))
