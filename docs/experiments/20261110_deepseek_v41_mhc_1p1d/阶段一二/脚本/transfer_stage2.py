"""Move the frozen runtime through AI Book; hashes on both hops, no model copy."""
from pathlib import Path
import sys, time, json, hashlib
BASE=Path(__file__).resolve().parents[1]
WORKSPACE=BASE.parents[2]
sys.path.insert(0,str(WORKSPACE/'sj9_12_登陆工具'))
import sj_access
cache=BASE/'传输缓存';remote='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/20261110_ops/stage2/'
records=[]
with sj_access.ssh_client('dev6') as c:
 s=c.open_sftp();expected=s.open(remote+'stage2_runtime.tar.gz.sha256').read().decode().split()[0]
 f=cache/'stage2_runtime.tar.gz';t=time.monotonic()
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:s.get(remote+f.name,str(f),prefetch=True)
 assert hashlib.sha256(f.read_bytes()).hexdigest()==expected
 print(json.dumps({'download':f.name,'seconds':time.monotonic()-t,'bytes':f.stat().st_size}),flush=True)
 s.close()
with sj_access.ssh_client('sj9') as c:
 s=c.open_sftp()
 for f in [cache/'source_stage2_mhc.tar.gz',cache/'stage2_runtime.tar.gz']:
  dest='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/incoming/'+f.name
  t=time.monotonic();s.put(str(f),dest+'.partial');s.rename(dest+'.partial',dest)
  record={'file':f.name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'upload_seconds':time.monotonic()-t}
  records.append(record);(BASE/'证据/stage2_transfer.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
 s.close()
