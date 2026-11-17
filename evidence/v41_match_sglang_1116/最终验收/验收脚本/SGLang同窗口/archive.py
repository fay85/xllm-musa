from pathlib import Path
import json,tarfile,hashlib,datetime
base=Path('/data/nfs_shared/xiaopang');goal=base/'v41_xllm_match_sglang_20261116';run=goal/'runs/sglang_B_recheck_v1';sg=base/'sglang_v41_reference_20261116/B_recheck_20261117'
assert json.loads((run/'finished.json').read_text())['exit_code']==0
assert json.loads((run/'performance/compilation_window.json').read_text())['all_clear']
for role in ['prefill','decode']:
 d=sg/'runtime'/role
 assert json.loads((d/'resource_release.json').read_text())['own_resources_released']
 assert json.loads((d/'environment.json').read_text())['git']=='7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6'
 assert json.loads((d/'rank0_loaded_libraries_sha.json').read_text())
p=goal/'exports/sglang_B_recheck_v1_evidence.tar.gz'
with tarfile.open(p,'x:gz',compresslevel=3) as t:
 t.add(run,arcname='goal_client')
 for name in ['configs','reference','runtime','client.py']:
  t.add(sg/name,arcname='sglang/'+name)
v={'archive':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()}
p.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
