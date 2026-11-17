from pathlib import Path
import json,hashlib,tarfile,datetime
r=Path('/expt');a=r/'exports/e50_gate_simt_checks_v1.tar.gz';assert not a.exists()
for name in ['e50_native_gate_gemv_gpu_v1','e50_production_dispatch_gpu_v1']:
 d=r/'diagnostics'/name;assert json.loads((d/'finished.json').read_text())['exit_code']==0;assert json.loads((d/'resource_release.json').read_text())['gpu_free']
with tarfile.open(a,'w:gz',compresslevel=3) as t:
 for name in ['e50_native_gate_gemv_gpu_v1','e50_production_dispatch_gpu_v1']:t.add(r/'diagnostics'/name,arcname='checks/'+name)
 for name in ['e50_native_gate_gemv_v1','e50_native_gate_gemv_v2','e50_native_gate_gemv_v3']:
  b=r/'builds'/name
  for p in b.iterdir():
   if p.is_dir() and p.name=='src' or p.is_file() and p.suffix in ['.json','.log','.py']:t.add(p,arcname='build/'+name+'/'+p.name)
 for name in ['validate_v41_gate_gemv_v1.py','validate_v41_gate_simt_dispatch_v1.py']:t.add(r/'scripts'/name,arcname='scripts/'+name)
 t.add(r/'inputs/e50_real_gate_weights.json',arcname='fixture_manifest.json')
v={'archive':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'bytes':a.stat().st_size,'at':datetime.datetime.now().astimezone().isoformat()};a.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
