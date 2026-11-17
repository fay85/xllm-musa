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


"""Host payload: start one previously owned frozen B role after fresh checks."""
from pathlib import Path
import datetime
import json
import socket
import subprocess
import time

host = {'worker32007': 'sj9', 'worker32009': 'sj10'}[socket.gethostname()]
role = 'prefill' if host == 'sj9' else 'decode'
name = f'xiaopang1116_sglang_{role}_{host}'
expected_cid = {'sj9': '034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2',
                'sj10': '4230b093ad6e326f5c30486a1d855573441bed6501d7cf6c436f6f14b0d0b55c'}[host]
root = Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116')
run = root / 'B_recheck_20261117'
control = Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control')
for peer in ['sj9', 'sj10']:
    check = json.loads((control / f'{peer}_preflight_p0.json').read_text())
    assert check['gpu_free'] and time.time() - datetime.datetime.fromisoformat(check['at']).timestamp() < 180
gpu = subprocess.run(['timeout', '20', 'mthreads-gmi'], capture_output=True, text=True, timeout=25)
assert gpu.returncode == 0 and 'No running processes found' in gpu.stdout
ports = {int(x.split()[3].rsplit(':', 1)[-1]) for x in subprocess.check_output(['ss', '-ltnH'], text=True).splitlines()}
assert not ports & ({28586, 28998, 28192, 29031} if host == 'sj9' else {28588})
container = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
assert container['Id'] == expected_cid
assert container['Image'] == 'sha256:1612b9c34896d7efb631975bf49542693c4c6d129554c63f32a31b13edc37337'
assert container['Config']['Labels']['owner'] == 'xiaopang'
assert container['Config']['Labels']['task'] == 'sglang-ab-trace-20261116'
out = run / 'runtime' / role
out.mkdir(parents=True, exist_ok=False)
(out / 'container_before.json').write_text(json.dumps(container, indent=2))
(out / 'gpu_before.txt').write_text(gpu.stdout)
if not container['State']['Running']:
    subprocess.run(['docker', 'start', name], check=True, capture_output=True, timeout=90)
payload = r"""
from pathlib import Path
import datetime,hashlib,importlib.metadata,json,os,subprocess,sys
role=sys.argv[1];r=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117')
out=r/'runtime'/role;script=r/'configs'/('B_'+role+'.sh')
git=subprocess.check_output(['git','-C','/workspace/sglang','rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C','/workspace/sglang','status','--porcelain'],text=True)
assert git=='7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6' and not dirty
versions={n:importlib.metadata.version(n) for n in ['torch','torch_musa','mate','mate-mubin','sglang-router','deep_ep']}
assert versions=={'torch':'2.9.0','torch_musa':'2.9.0','mate':'0.2.6','mate-mubin':'0.2.6','sglang-router':'0.2.4','deep_ep':'1.1.0+9017268.musa438.ringreset1'}
expected=json.loads((r/'reference'/('core_libraries_'+role+'.json')).read_text());actual=[]
for row in expected:
 p=Path(row['path']);assert p.is_file(),str(p)
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()
 assert h==row['sha256'],str(p)
 actual.append({'path':str(p),'sha256':h})
(out/'environment.json').write_text(json.dumps({'git':git,'git_status':dirty,'versions':versions,'libraries':actual},indent=2))
with (out/'server.log').open('xb') as log:
 p=subprocess.Popen(['bash',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record={'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'script':str(script),'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
(out/'launcher.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
"""
result = subprocess.run(['docker', 'exec', '-i', name, '/opt/venv/bin/python', '-', role],
                        input=payload, text=True, capture_output=True, timeout=90)
(out / 'launch_result.json').write_text(json.dumps({'exit': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}, indent=2))
print(result.stdout); print(result.stderr)
assert result.returncode == 0

