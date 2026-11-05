"""Read-only final archive; retain this task's single stopped container per host."""
import time;time.sleep(1)
from pathlib import Path
import argparse,datetime,json,subprocess
parser=argparse.ArgumentParser();parser.add_argument('--host',choices=['dev4','dev7'],required=True);args=parser.parse_args();host=args.host
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离');run='pd_dev4_dev7_v2_nic9';out=root/'runs'/run/host
name=f'xiaopang1105_dsv41_{"prefill" if host=="dev4" else "decode"}_{host}_{run}'
docker=['docker']+(['-H','unix:///run/xiaopang1105-docker-dev4.sock'] if host=='dev4' else [])
result=json.loads((out/'result.json').read_text());assert result['exit_codes']==[0]*8,result
inspect=json.loads(subprocess.check_output(docker+['inspect',name],text=True))[0]
assert not inspect['State']['Running'] and inspect['State']['ExitCode']==0,inspect['State']
(out/'container_final_inspect.json').write_text(json.dumps(inspect,indent=2))
gpu=subprocess.check_output(['mthreads-gmi'],text=True);(out/'gpu_after_stop.txt').write_text(gpu)
receipt={'at':datetime.datetime.now().astimezone().isoformat(),'host':host,'container':name,'container_state':inspect['State'],'rank_exit_codes':result['exit_codes'],'container_retained':True,'gpu_report':'gpu_after_stop.txt'}
(out/'final_archive.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));print(gpu)
