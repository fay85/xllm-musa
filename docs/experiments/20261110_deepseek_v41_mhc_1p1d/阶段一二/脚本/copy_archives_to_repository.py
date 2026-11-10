"""Copy verified experiment archives into the report, without model weights."""
from pathlib import Path
import sys,json,hashlib
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
repo=b.parents[1]/'20261105跑通PD分离/代码/xllm-musa';out=repo/'docs/experiments/20261110_deepseek_v41_mhc_1p1d/raw_runs';out.mkdir(parents=True,exist_ok=True);records=[]
with sj_access.ssh_client('sj10') as c:
 with c.open_sftp() as f:
  for p in sorted((b/'证据').glob('*_archive_receipt.json')):
   d=json.loads(p.read_text());target=out/(d.get('archive_name',d['run'])+'.tar.gz')
   if not target.exists():f.get(d['path'],str(target),prefetch=False)
   assert target.stat().st_size==d['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==d['sha256'],str(target)
   records.append({'file':target.name,'sha256':d['sha256'],'bytes':d['bytes'],'validation_status':d.get('validation_status','PASS'),'remote_archive':d['path'],'snapshot_verified_at':d['verified_at']})
for record in records:
 if record['file']=='stage1_ref.tar.gz':record['preferred_evidence_archive']='stage1_ref_proofs.tar.gz';record['note']='Earlier historical snapshot. Performance data are valid; use the proofs archive for all 16 immutable PD log snapshots and canonical request hashes.'
 if record['validation_status']=='FAIL':record['note']='Failed experiment retained for diagnosis; never use as a valid performance sample.'
(out/'manifest.json').write_text(json.dumps({'archives':records,'scope':'Historical verified snapshots; later derived summary annotations may be stored separately. Raw SQLite and logs are preserved. No model weights or binaries are bundled.'},indent=2));print(json.dumps(records))
