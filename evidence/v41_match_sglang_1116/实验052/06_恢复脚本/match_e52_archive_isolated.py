from pathlib import Path
import tarfile,json,hashlib,datetime
r=Path('/expt');z=r/'exports/e52_isolated_checks_v1.tar.gz';assert not z.exists();dirs=[r/'diagnostics'/n for n in ['e52_rope_table_gpu_v1','e52_rope_table_gpu_v2','e52_rope_arithmetic_gpu_v3','e52_rope_table_gpu_v4']]+[r/'builds'/n for n in ['e52_rope_table_module_v1','e52_rope_table_module_v2']]
for d in dirs:assert (d/'finished.json').exists()
with tarfile.open(z,'w:gz') as t:
 for d in dirs:
  for p in d.rglob('*'):
   if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.log','.py','.cpp','.cu','.txt']:t.add(p,arcname=str(p.relative_to(r)))
 for n in ['validate_v41_rope_table_v1.py','validate_v41_rope_table_v2.py','validate_v41_rope_arithmetic_v3.py','validate_v41_rope_table_v4.py']:
  p=r/'scripts'/n;t.add(p,arcname='scripts/'+n)
v={'archive':str(z),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()};z.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
