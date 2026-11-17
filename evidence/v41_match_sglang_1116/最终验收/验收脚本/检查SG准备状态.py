from pathlib import Path
import json,hashlib
r=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117');v={'exists':r.exists()}
for name in ['client.py','configs/B_prefill.sh','configs/B_decode.sh']:
 p=r/name;v[name]={'exists':p.exists(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}
v['runtime_exists']=(r/'runtime').exists();print(json.dumps(v))
