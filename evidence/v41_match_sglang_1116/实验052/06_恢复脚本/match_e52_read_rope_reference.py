from pathlib import Path
import json,hashlib,importlib.util,tarfile
spec=importlib.util.find_spec('sglang');roots=list(spec.submodule_search_locations);p=Path(roots[0]);files=['srt/hardware_backend/layers/deepseek_v4_musa/ops/norm_rope_ops.py','kernels/ops/attention/deepseek_v4_rope.py']
o=Path('/expt/diagnostics/e52_rope_reference');o.mkdir(exist_ok=False);rows=[]
for name in files:
 f=p/name
 if f.exists():
  s=f.read_text();dest=o/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(s);rows.append({'file':str(f),'relative':name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'hits':[{'line':i,'text':l} for i,l in enumerate(s.splitlines(),1) if any(k in l for k in ['def ','import','@triton','@tilelang'])]})
(o/'manifest.json').write_text(json.dumps(rows,indent=2));z=Path('/expt/exports/e52_rope_reference.tar.gz')
with tarfile.open(z,'w:gz') as t:t.add(o,arcname='e52_rope_reference')
print(json.dumps({'archive':str(z),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'manifest':rows}))
