from pathlib import Path
import json,hashlib,tarfile,datetime
r=Path('/expt');run=r/'runs/p41_static_activation_proposal3_reverse_v1';assert json.loads((run/'formal_client/finished.json').read_text())['exit_code']==0
for h in ['sj9','sj10']:assert json.loads((run/h/'result.json').read_text())['exit_codes']==[0]*8
p=r/'exports/p41_static_activation_proposal3_reverse_v1_evidence.tar.gz';assert not p.exists()
with tarfile.open(p,'w:gz',compresslevel=3) as t:
 t.add(run/'formal_client',arcname='formal_client')
 for h in ['sj9','sj10']:
  for n in ['role_config.json','runtime_identity.json','status.json','result.json','rank0.log','supervisor_job.json']:t.add(run/h/n,arcname='roles/'+h+'/'+n)
v={'archive':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()};p.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
