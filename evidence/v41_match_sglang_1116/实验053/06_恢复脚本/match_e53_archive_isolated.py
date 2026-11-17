from pathlib import Path
import tarfile,json,hashlib,datetime
r=Path('/expt');z=r/'exports/e53_isolated_checks_v1.tar.gz';assert not z.exists();dirs=[r/'diagnostics/e53_rope_qdq_gpu_v1',r/'builds/e53_rope_qdq_module_v1']
for d in dirs:assert json.loads((d/'finished.json').read_text())['exit_code']==0
with tarfile.open(z,'w:gz') as t:
 for d in dirs:
  for p in d.rglob('*'):
   if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.log','.py','.cpp','.cu','.txt']:t.add(p,arcname=str(p.relative_to(r)))
 p=r/'scripts/validate_v41_rope_qdq_v1.py';t.add(p,arcname='scripts/'+p.name)
v={'archive':str(z),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()};z.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
