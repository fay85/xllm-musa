"""Read-only final resource receipt for the two explicitly owned containers."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import shlex
import sys

base = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base.parents[2] / 'sj9_12_登陆工具'))
import sj_access

state = json.loads((base / '证据/matrix_status.json').read_text())
assert state['phase'] == 'ALL_BENCHMARKS_VALIDATED_AND_RANKS_STOPPED', state
code = '''from pathlib import Path
import subprocess,json,re
host=HOST
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
name='xiaopang1110_v41_'+('prefill' if host=='sj9' else 'decode')+'_'+host
result=json.loads((root/'runs/stage2_all'/host/'result.json').read_text())
assert result['exit_codes']==[0]*8 and result.get('control_exit') in (None,0)
gmi=subprocess.check_output(['mthreads-gmi'],text=True)
cards=re.findall(r'(\\d+)%\\s+(\\d+)MiB\\((\\d+)MiB\\)',gmi)
top=subprocess.check_output(['docker','top',name,'-eo','pid,stat,comm,args'],text=True)
process_lines=top.splitlines()[1:]
assert not any('/work/artifacts/stage2_mhc/xllm' in line or '/opt/xllm_service/bin/xllm_master_serving' in line for line in process_lines),top
container_state=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .State}}',name],text=True))
print(json.dumps({'host':host,'container':name,'state':container_state,'top':top,'last_run_result':result,'all_eight_gpus_empty':len(cards)==8 and all(u=='0' and m=='0' for u,m,_ in cards),'gmi':gmi}))
'''

def read_host(host: str) -> dict:
    with sj_access.ssh_client(host) as client:
        result = sj_access.execute(client, host, 'python3 -c ' + shlex.quote(code.replace('HOST', repr(host))), root=True, timeout=30)
    assert result['exit_status'] == 0, result
    return json.loads(result['stdout'])

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    hosts = list(pool.map(read_host, ('sj9', 'sj10')))
receipt = {'at': datetime.datetime.now().astimezone().isoformat(), 'status': 'OWN_MODEL_AND_CONTROL_PROCESSES_EXITED_NORMALLY', 'hosts': hosts}
(base / '证据/final_resource_release.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
print(json.dumps({'at': receipt['at'], 'status': receipt['status'], 'hosts': [{k: h[k] for k in ('host','container','all_eight_gpus_empty')} for h in hosts]}, ensure_ascii=False))
