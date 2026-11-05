"""Normal TERM through this run's supervisors after both benchmark groups finish."""
import time;time.sleep(1)
from pathlib import Path
import datetime,json,sqlite3
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/runs/pd_dev4_dev7_v2_nic9')
perf=root/'performance_v2'
assert (perf/'client_runs_finished.json').is_file(),'Client still running'
status=json.loads((perf/'status.json').read_text())
assert len(status)==4 and all(x['exit_code']==0 for x in status),status
for length in [256,2048]:
 databases=list((perf/f'input{length}_output256'/'measured').rglob('benchmark_data.db'))
 assert len(databases)==1
 with sqlite3.connect('file:'+str(databases[0])+'?mode=ro',uri=True) as c:
  rows=c.execute('SELECT success,prompt_tokens,completion_tokens FROM result').fetchall()
 assert len(rows)==5 and all(r==(1,length,256) for r in rows),rows
receipt={'at':datetime.datetime.now().astimezone().isoformat(),'method':'one normal TERM via own supervisor stop.request','requests':[]}
for host in ['dev4','dev7']:
 p=root/host/'stop.request'
 if not p.exists():
  with p.open('x') as f:f.write('Normal stop after two completed benchmark groups: '+receipt['at'])
 receipt['requests'].append({'host':host,'path':str(p)})
(root/'shutdown_requested.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
