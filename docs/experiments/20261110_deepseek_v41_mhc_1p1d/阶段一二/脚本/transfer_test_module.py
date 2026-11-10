from pathlib import Path
import sys,shlex,json,hashlib
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
d='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/20261110_ops/stage2';s='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops';filename='musa_python_kernel_test_module.cpython-310-x86_64-linux-gnu.so';archive=b/'传输缓存/test_module.tar.gz'
with sj_access.ssh_client('dev6') as c:
 code=f'from pathlib import Path;import tarfile,hashlib,json;p=Path({d!r});f=p/"test_module.tar.gz";t=tarfile.open(f,"w:gz",compresslevel=3);t.add(p/"artifacts"/{filename!r},arcname={filename!r});t.close();print(json.dumps({{"archive_sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"module_sha256":hashlib.sha256((p/"artifacts"/{filename!r}).read_bytes()).hexdigest()}}))'
 r=sj_access.execute(c,'dev6','python3 -c '+shlex.quote(code),root=True,timeout=35);assert r['exit_status']==0;manifest=json.loads(r['stdout'])
 with c.open_sftp() as f:f.get(d+'/test_module.tar.gz',str(archive))
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive_sha256']
with sj_access.ssh_client('sj9') as c:
 with c.open_sftp() as f:f.put(str(archive),s+'/incoming/test_module.tar.gz')
 code=f'from pathlib import Path;import tarfile,hashlib,json;p=Path({s!r});f=p/"incoming/test_module.tar.gz";assert hashlib.sha256(f.read_bytes()).hexdigest()=={manifest["archive_sha256"]!r};out=p/"artifacts/test_module";out.mkdir(exist_ok=False);t=tarfile.open(f);t.extractall(out);assert hashlib.sha256((out/{filename!r}).read_bytes()).hexdigest()=={manifest["module_sha256"]!r};(p/"00_manifest/test_module.json").write_text(json.dumps({manifest!r},indent=2));print("test extension verified")'
 r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(code),root=True,timeout=30);assert r['exit_status']==0;print(r)
 rpath=Path('王家明/20261105跑通PD分离/代码/xllm-musa/tools/musa_deepseek_v41/validate_hyper_connection.py')
 r=sj_access.execute(c,'sj9','tee '+shlex.quote(s+'/scripts/validate_hyper_connection_final.py')+' >/dev/null',root=True,payload=rpath.read_bytes());assert r['exit_status']==0
(b/'证据/test_module_transfer.json').write_text(json.dumps(manifest|{'archive_bytes':archive.stat().st_size},indent=2));archive.unlink();print('Local intermediary removed after verification')
