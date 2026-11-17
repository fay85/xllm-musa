"""Host CPU payload, after SG roles passed frozen environment checks."""
from pathlib import Path
import hashlib,json
sg=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117')
goal=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116')
env={role:json.loads((sg/'runtime'/role/'environment.json').read_text()) for role in ['prefill','decode']}
assert all(v['git']=='7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6' and v['versions']['mate']=='0.2.6' for v in env.values())
client=goal/'scripts/sglang_reference_final_client.py'
if client.exists():assert client.read_bytes()==(sg/'client.py').read_bytes()
else:
 with client.open('x') as f:f.write((sg/'client.py').read_text())
identity={'image_id':'sha256:1612b9c34896d7efb631975bf49542693c4c6d129554c63f32a31b13edc37337','sglang_git':'7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6','case':'B','client_host':'sj10','source_reference':str(sg),'environment':env,'config_shas':{role:hashlib.sha256((sg/'configs'/('B_'+role+'.sh')).read_bytes()).hexdigest() for role in ['prefill','decode']},'engram':'Frozen GPU Engram short-request B; CP8/DP8/DSPARK5/full graph'}
idpath=goal/'scripts/sglang_reference_final_identity.json'
if idpath.exists():assert json.loads(idpath.read_text())==identity
else:
 with idpath.open('x') as f:json.dump(identity,f,indent=2)
print(json.dumps({'client_published':str(client),'gpu_started':False}))

