from pathlib import Path
import tarfile,json,hashlib,datetime
r=Path('/expt');b=r/'diagnostics/e56_short_context_gpu_v1';assert json.loads((b/'finished.json').read_text())['exit_code']==0
archive=r/'exports/e56_checks_gpu_v1.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for f in b.rglob('*'):
  if f.is_file() and f.suffix in ['.json','.jsonl','.log','.py','.cpp']:t.add(f,arcname=str(f.relative_to(b)))
v={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()};print(json.dumps(v))
