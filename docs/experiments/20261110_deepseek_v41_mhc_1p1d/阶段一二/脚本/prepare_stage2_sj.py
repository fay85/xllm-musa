from pathlib import Path
import json,sys,shlex,base64
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
code=r'''from pathlib import Path
import json,sys,hashlib,subprocess
x=json.load(sys.stdin);root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
for item in x['transfers']:
 f=root/'incoming'/item['file'];assert f.stat().st_size==item['bytes'];assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
repo=root/'01_sources/stage2_mhc';repo.mkdir(exist_ok=False);binary=root/'artifacts/stage2_mhc';binary.mkdir(exist_ok=False)
subprocess.run(['tar','-xzf',str(root/'incoming/source_stage2_mhc.tar.gz'),'-C',str(repo)],check=True)
subprocess.run(['tar','-xzf',str(root/'incoming/stage2_runtime.tar.gz'),'-C',str(binary)],check=True)
for name,sha in x['source']['changed_files'].items():assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==sha
x['binary_sha256']=hashlib.sha256((binary/'xllm').read_bytes()).hexdigest();x['binary_bytes']=(binary/'xllm').stat().st_size
(root/'00_manifest/stage2_source.json').write_text(json.dumps(x,indent=2));print(json.dumps({'binary_sha256':x['binary_sha256'],'binary_bytes':x['binary_bytes'],'source':str(repo)}))
# Benchmark binary gate is explicit. The currently running stage1 process has
# already parsed its old default; no model code or running configuration changes.
p=root/'scripts/run_official_performance.py';s=p.read_text();s=s.replace("args=parser.parse_args()","parser.add_argument('--binary',type=Path,default=Path('/work/artifacts/bin/xllm'));args=parser.parse_args()",1).replace("binary=root/'artifacts/bin/xllm'","binary=args.binary");p.write_text(s)
'''
with sj_access.ssh_client('sj9') as c:
 r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(code),root=True,payload=json.dumps({'transfers':json.loads((b/'证据/stage2_transfer.json').read_text()),'source':json.loads((b/'证据/stage2_source_manifest.json').read_text())}),timeout=60)
 (b/'证据/stage2_prepare_sj.json').write_text(json.dumps(r,indent=2));print(r);assert r['exit_status']==0
