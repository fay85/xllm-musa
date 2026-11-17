from pathlib import Path
import json
r=Path('/expt/builds/e50_native_gate_gemv_v2')
z={}
for name in ['job.json','child.json','finished.json','worker.log','0.log','1.log','import.log']:
 p=r/name
 if p.exists():z[name]=p.read_text()[-5000:]
print(json.dumps(z))

p=Path('/expt/inputs/e50_real_gate_weights.json')
print('FIXTURE',p.read_text() if p.exists() else 'missing')
