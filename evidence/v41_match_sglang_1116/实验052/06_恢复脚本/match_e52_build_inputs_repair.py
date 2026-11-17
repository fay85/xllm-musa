from pathlib import Path
import json,tarfile,hashlib
s=Path('/workspace/sglang/python/sglang');o=Path('/expt/diagnostics/e52_rope_build_inputs');assert (o/'rope_compile_command.json').exists();paths=list(s.rglob('*norm_rope*'));print(json.dumps({'matches':[str(x) for x in paths]}))
for p in paths:
 if p.is_file() and p.suffix=='.py' and 'kernel' in p.name:(o/p.name).write_bytes(p.read_bytes())
z=Path('/expt/exports/e52_rope_build_inputs.tar.gz');assert not z.exists()
with tarfile.open(z,'w:gz') as t:t.add(o,arcname='e52_rope_build_inputs')
print(json.dumps({'archive':str(z),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'compile':json.loads((o/'rope_compile_command.json').read_text())}))
