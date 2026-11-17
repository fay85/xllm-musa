from pathlib import Path
import json
r=Path('/expt/builds/e50_native_gate_simt_v1');v={}
for n in ['job.json','finished.json']:
 p=r/n
 if p.exists():v[n]=json.loads(p.read_text())
p=r/'build.log';s=p.read_text(errors='replace') if p.exists() else '';v['tail']=s[-1600:];v['errors']=[x for x in s.splitlines() if ' error:' in x or 'FAILED:' in x];print(json.dumps(v))
