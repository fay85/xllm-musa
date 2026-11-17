import runpy as _compile_guard_runpy
from pathlib import Path as _compile_guard_path
_compile_guard_root = _compile_guard_path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
if not _compile_guard_root.exists():
    _compile_guard_root = _compile_guard_path('/expt/00_control')
_compile_guard_runpy.run_path(str(_compile_guard_root / 'assert_compilation_clear.py'))

# Mandatory shared-resource cooldown before any GPU resource query or launch.
from pathlib import Path as _CooldownPath
import datetime as _cooldown_datetime
import json as _cooldown_json
_cooldown_path = _CooldownPath('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control/external_release_cooldown.json')
if _cooldown_path.exists():
    _cooldown = _cooldown_json.loads(_cooldown_path.read_text())
    if _cooldown.get('active') and _cooldown_datetime.datetime.now().astimezone() < _cooldown_datetime.datetime.fromisoformat(_cooldown['not_before']):
        raise SystemExit('External-use cooldown: no GPU polling or launch before ' + _cooldown['not_before'])


from pathlib import Path
import json,subprocess,datetime,socket,time
h='sj9' if socket.gethostname()=='worker32007' else 'sj10';names={'sj9':('xiaopang1117_v41_sg438_prefill_sj9','52abeddab774ff06522e2ffe452b0714065c579f95137660f51f54ed3d9ec94a'),'sj10':('xiaopang1117_v41_sg438_decode_sj10','7eefcf3aedcb80825833e6b47295689965905518e51823cab8bedecfad6374f3')};name,cid=names[h];c=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert c['Id']==cid and c['State']['Running'];r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116')
assert json.loads((r/'diagnostics/e46_native_mhc_dispatch_v2/finished.json').read_text())['exit_code']==0
assert json.loads((r/'diagnostics/e51_gather_gpu_v2_bits/finished.json').read_text())['special_contract_pass']
assert json.loads((r/'diagnostics/e52_rope_production_gpu_v1/finished.json').read_text())['exit_code']==0
checks=r/'builds/native_sg438_control'
assert json.loads((checks/'operator_gpu_v3/finished.json').read_text())['all_pass']
assert json.loads((r/'diagnostics/e37_native_sync_mccl/runtime_v1/finished.json').read_text())['all_pass']
assert all(v['exit_code']==0 for v in json.loads((checks/'operator_gpu_v2/finished.json').read_text())['results'][:2])
for host in ['sj9','sj10']:
 p=json.loads((r/'00_control'/f'{host}_preflight_p0.json').read_text());assert p['gpu_free'];assert time.time()-datetime.datetime.fromisoformat(p['at']).timestamp()<180,'Stale resource preflight'
g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True,timeout=25);assert g.returncode==0 and 'No running processes found' in g.stdout,'GPU changed since preflight'
ports=subprocess.check_output(['ss','-ltnH'],text=True);used={int(x.split()[3].rsplit(':',1)[-1]) for x in ports.splitlines()};needed={22000,22100,22200,22300} if h=='sj9' else {32000,32100,32200,32300,31178,31179,31180,31181,31182,31183,35690};assert not used&needed,used&needed
payload="""from pathlib import Path
import subprocess,json,os,datetime
r=Path('/expt/runs/final_long_v1')/HOST
assert (r/'prepared_config.json').exists()
assert not (r/'supervisor_job.json').exists() and not (r/'rank0.log').exists()
with (r/'supervisor.log').open('xb') as f:
 p=subprocess.Popen(['python3','/expt/scripts/pd_rank_supervisor_nocore_v1.py','--config',str(r/'prepared_config.json'),'--output',str(r)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc') .joinpath(str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'host':HOST};(r/'supervisor_job.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
""".replace('HOST',repr(h))
p=subprocess.run(['docker','exec','-i',name,'python3','-'],input=payload,text=True,capture_output=True,timeout=30);print(p.stdout);print(p.stderr);assert p.returncode==0

