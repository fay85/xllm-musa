from pathlib import Path
import sys,json,base64,shlex
BASE=Path(__file__).resolve().parents[1];W=BASE.parents[2];OLD=BASE.parents[1]/'20261105跑通PD分离/脚本'
sys.path.insert(0,str(W/'sj9_12_登陆工具'));import sj_access
files={}
for name in ['pd_rank_supervisor.py','pd_control_supervisor.py','check_basic_questions.py','prepare_benchmark_inputs.py','run_official_performance.py','validate_engram_sidecar.py']:
 s=(OLD/name).read_text().replace('deepseek-v4.1-flash-0909','deepseek-v4.1-flash-1017').replace('model_view_0909','model_view_1017')
 files['scripts/'+name]=base64.b64encode(s.encode()).decode()
code=r'''from pathlib import Path
import sys,json,base64,hashlib,subprocess,shutil
x=json.load(sys.stdin);root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
for item in x['transfers']:
 p=root/'incoming'/item['file'];assert p.stat().st_size==item['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
repo=root/'01_sources/34815f6a';repo.mkdir(exist_ok=False)
subprocess.run(['tar','-xzf',str(root/'incoming/source_stage1_34815f6a.tar.gz'),'-C',str(repo)],check=True)
subprocess.run(['tar','-xzf',str(root/'incoming/stage1_runtime.tar.gz'),'-C',str(root)],check=True)
(root/'repos').mkdir(exist_ok=True);(root/'repos/xllm-musa').symlink_to('../01_sources/34815f6a')
for n,v in x['files'].items():(root/n).write_bytes(base64.b64decode(v))
source=Path('/data/nfs_shared/models/DeepSeek-V4.1-Flash-1017');view=root/'artifacts/model_view_1017';view.mkdir(exist_ok=False);records=[]
for p in sorted(source.rglob('*')):
 rel=p.relative_to(source)
 if any(t.startswith('.') or t=='__pycache__' for t in rel.parts) or not p.is_file():continue
 q=view/rel;q.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix=='.safetensors':q.symlink_to(p);records.append({'path':str(rel),'bytes':p.stat().st_size,'kind':'weight_symlink'})
 else:
  assert p.stat().st_size<200*1024**2
  shutil.copy2(p,q);records.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
(root/'00_manifest/model_view.json').write_text(json.dumps({'source':str(source),'view':str(view),'files':records},indent=2))
assert hashlib.sha256((root/'artifacts/bin/xllm').read_bytes()).hexdigest()=='8d5e8c5e83227021672a472f749782b4426654abf264f10f815923e57074caa6'
(root/'00_manifest/stage1_source.json').write_text(json.dumps({'commit':'34815f6ac23de6c3145a3392b7649874a37d4f27','archive':x['transfers'],'repo':str(repo)},indent=2))
(root/'evidence').mkdir(exist_ok=True)
print(json.dumps({'source':str(repo),'model_view':str(view),'files':len(records)}))
'''
with sj_access.ssh_client('sj9') as c:
 r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(code),root=True,payload=json.dumps({'files':files,'transfers':json.loads((BASE/'证据/stage1_transfer.json').read_text())}),timeout=55);(BASE/'证据/stage1_prepare_receipt.json').write_text(json.dumps(r,indent=2));print(r);assert r['exit_status']==0
 cmd=['docker','exec','-d','-e','PYTHONPATH=/work/repos/xllm-musa','-e','MUSA_VISIBLE_DEVICES=-1','xiaopang1110_v41_prefill_sj9','bash','-c','cd /work/repos/xllm-musa; python3 /work/scripts/validate_engram_sidecar.py > /work/evidence/engram_hash.log 2>&1; echo $? > /work/evidence/engram_hash.exit']
 print(sj_access.execute(c,'sj9',shlex.join(cmd),root=True,timeout=15))
 print(sj_access.execute(c,'sj9',"find /data/nfs_shared/xiaopang -maxdepth 7 -type f -name evalscope -path '*/bin/*' -print",root=True,timeout=35))
