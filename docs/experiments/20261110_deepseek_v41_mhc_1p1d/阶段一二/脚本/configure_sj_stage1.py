from pathlib import Path
import sys,shlex,json,base64
BASE=Path(__file__).resolve().parents[1];W=BASE.parents[2];OLD=BASE.parents[1]/'20261105跑通PD分离/脚本'
sys.path.insert(0,str(W/'sj9_12_登陆工具'));import sj_access
s=(OLD/'prepare_role_configs.py').read_text()
s=s.replace("ips={'dev4':'10.20.34.24','dev5':'10.20.34.25','dev6':'10.20.34.26','dev7':'10.20.34.28'}","ips={'sj9':'10.121.32.7','sj10':'10.121.32.9'}")
s=s.replace('model_view_0909','model_view_1017').replace('deepseek-v4.1-flash-0909','deepseek-v4.1-flash-1017').replace("'PREFILL',21000","'PREFILL',31000").replace("'DECODE',22000","'DECODE',32000").replace('2907','3907').replace('2908','3908').replace('xiaopang1105-dsv41','xiaopang1110-dsv41')
script=r'''from pathlib import Path
import json,sys,base64,shutil,subprocess
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
(root/'scripts/prepare_role_configs.py').write_bytes(base64.b64decode(json.load(sys.stdin)['code']))
a=Path('/data/nfs_shared/xiaopang/20261104benchmark1P1D/eval/venv');b=root/'artifacts/bench_venv_v2'
if not b.exists():shutil.copytree(a,b,symlinks=True)
p=b/'bin/evalscope';text=p.read_text();text=text.replace(text.splitlines()[0],'#!/work/artifacts/bench_venv_v2/bin/python');p.write_text(text)
name='xiaopang1110_v41_prefill_sj9'
for args in [ ['python3','/work/scripts/prepare_role_configs.py','--prefill','sj9','--decode','sj10','--run-id','stage1_ref'],['python3','/work/scripts/prepare_benchmark_inputs.py'] ]:
 r=subprocess.run(['docker','exec','-e','MUSA_VISIBLE_DEVICES=-1',name]+args,capture_output=True,text=True,timeout=45);print(r.stdout,r.stderr);r.check_returncode()
print('prepared')
'''
with sj_access.ssh_client('sj9') as c:
 r=sj_access.execute(c,'sj9','python3 -c '+shlex.quote(script),root=True,payload=json.dumps({'code':base64.b64encode(s.encode()).decode()}),timeout=120);(BASE/'证据/configure_stage1.json').write_text(json.dumps(r,indent=2));print(r);assert r['exit_status']==0
 for host in ['sj9','sj10']:
  result=json.loads((BASE/'证据'/f'{host}_nics.json').read_text());data=json.loads(result['stdout'].splitlines()[0]);nics=','.join(v['nic'] for v in data['nics'] if v['state']=='4: ACTIVE' and v['rate'].startswith('400 '))
  patch="""from pathlib import Path
import json
p=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/artifacts/prepared/stage1_ref/HOST.json');d=json.loads(p.read_text())
for r in d['ranks']:
 r['environment'].update(MC_TE_FILTERS=NICS,MCCL_SOCKET_IFNAME='bond0',GLOO_SOCKET_IFNAME='bond0',XLLM_DSV41_MUSA_SHARED_SWIGLU='reference')
p.write_text(json.dumps(d,indent=2));print('prepared HOST NICs',NICS)
""".replace('HOST',host).replace('NICS',repr(nics))
  print(sj_access.execute(c,'sj9','python3 -c '+shlex.quote(patch),root=True,timeout=15))
