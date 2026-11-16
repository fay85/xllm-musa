from pathlib import Path
import subprocess,json,os,datetime
r=Path('/expt/runs/p3_attention_replica_graph_v1');assert not (r/'client_launch.json').exists();assert not (r/'client').exists();env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1',PYTHONPATH='',TOKENIZERS_PARALLELISM='false')
with (r/'client_supervisor.log').open('xb') as f:p=subprocess.Popen(['python3','/expt/scripts/match_p3_client.py','--run','p3_attention_replica_graph_v1'],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'client_launch.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
