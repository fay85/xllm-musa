from pathlib import Path
import json,tarfile,hashlib,datetime
r=Path('/expt');z=r/'exports/e51_builds_and_scripts.tar.gz';assert not z.exists()
with tarfile.open(z,'w:gz',compresslevel=3) as t:
 for d in [r/'builds'/n for n in ['e51_gather_module_v1','e51_gather_module_v2','e51_gather_module_v3','e51_gather_module_v4_bits','e51_native_bit_gather_v1']]:
  if not d.exists():continue
  for p in d.rglob('*'):
   if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.log','.py','.h','.cpp','.txt'] and p.stat().st_size<3000000:t.add(p,arcname=str(p.relative_to(r)))
 for p in (r/'scripts').glob('*e51*.py'):t.add(p,arcname='scripts/'+p.name)
 for name in ['validate_v41_gather_candidate.py','validate_v41_gather_bits_v2.py']:
  p=r/'scripts'/name
  if p.exists():t.add(p,arcname='scripts/'+p.name)
v={'archive':str(z),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()};z.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
