from pathlib import Path
import json,subprocess,os,datetime
root=Path('/expt');run=root/'runs/p48_runtime_boundaries_v1';assert not (run/'formal_client').exists();p=root/'scripts/match_e48_timing_client_v1.py';assert p.exists()
with (run/'formal_client_supervisor.log').open('xb') as log:q=subprocess.Popen(['python3',str(p),'--run',run.name],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1',PYTHONPATH='',TOKENIZERS_PARALLELISM='false'),start_new_session=True)
v={'pid':q.pid,'start_ticks':Path('/proc',str(q.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(run/'formal_client_launch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
