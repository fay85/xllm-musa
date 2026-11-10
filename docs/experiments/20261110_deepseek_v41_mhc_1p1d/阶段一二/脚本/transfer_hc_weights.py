from pathlib import Path
import sys,shlex,json,hashlib
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
s='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/artifacts/hc_1017_subset'
d='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/20261110_ops/stage2/artifacts/hc_1017_subset'
local=b/'传输缓存/hc_1017_subset';local.mkdir(exist_ok=True)
with sj_access.ssh_client('sj9') as c:
 with c.open_sftp() as f:
  for name in ['manifest.json','model.safetensors.index.json','hc.safetensors']:
   dest=local/name
   if name!='hc.safetensors':f.get(s+'/'+name,str(dest));continue
   size=f.stat(s+'/'+name).st_size;offset=dest.stat().st_size if dest.exists() else 0
   with f.open(s+'/'+name,'rb') as remote,dest.open('ab') as target:
    while offset<size:
     chunks=[(i,min(32768,size-i)) for i in range(offset,min(size,offset+16*32768),32768)]
     for data in remote.readv(chunks):target.write(data);offset+=len(data)
     target.flush();print('Downloaded',offset,'of',size,flush=True)
m=json.loads((local/'manifest.json').read_text());assert hashlib.sha256((local/'hc.safetensors').read_bytes()).hexdigest()==m['sha256']
with sj_access.ssh_client('dev6') as c:
 r=sj_access.execute(c,'dev6','mkdir -p '+shlex.quote(d),root=True);assert r['exit_status']==0
 for f in local.iterdir():
  r=sj_access.execute(c,'dev6','tee '+shlex.quote(d+'/'+f.name)+' >/dev/null',root=True,payload=f.read_bytes(),timeout=60);assert r['exit_status']==0
 r=sj_access.execute(c,'dev6','sha256sum '+shlex.quote(d+'/hc.safetensors'),root=True);assert m['sha256'] in r['stdout'];print(r)
(b/'证据/hc_1017_subset_manifest.json').write_text(json.dumps(m,indent=2))
