"""Host CPU payload: archive one completed final round, preserving original bytes."""
from pathlib import Path
import argparse,datetime,hashlib,json,tarfile
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,choices=['final_short1_v1','final_short2_v1','final_short3_v1','final_long_v1']);args=ap.parse_args()
root=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116');run=root/'runs'/args.run
assert json.loads((run/'formal_client/finished.json').read_text())['exit_code']==0
for host in ['sj9','sj10']:
 assert json.loads((run/host/'resource_release.json').read_text())['own_resources_released']
 assert json.loads((run/host/'result.json').read_text())['exit_codes']==[0]*8
target=root/'exports'/(args.run+'_evidence.tar.gz')
with tarfile.open(target,'x:gz',compresslevel=3) as t:
 for n in ['formal_client','experiment.json','formal_client_launch.json']:t.add(run/n,arcname=n)
 for host in ['sj9','sj10']:t.add(run/host,arcname='roles/'+host)
v={'archive':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'at':datetime.datetime.now().astimezone().isoformat()}
target.with_suffix('.receipt.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))

