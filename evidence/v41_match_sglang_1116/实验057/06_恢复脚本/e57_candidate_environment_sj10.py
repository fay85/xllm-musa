from pathlib import Path
import json,hashlib,datetime
r=Path('/expt/runs/p57_progress_disabled_v1');host='sj10';d=r/host
cfg=json.loads((d/'role_config.json').read_text());state=json.loads((d/'status.json').read_text());assert state['exit_codes']==[None]*8
rows=[]
for rank,pid in enumerate(state['pids']):
 p=Path('/proc',str(pid));assert hashlib.sha256((p/'exe').read_bytes()).hexdigest()==cfg['binary_sha256']
 env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
 enabled=b'XLLM_DSV41_DIAGNOSTIC_PROGRESS' in env;assert enabled==False
 rows.append({'rank':rank,'pid':pid,'start_ticks':(p/'stat').read_text().split()[21],'progress_present':enabled})
v={'at':datetime.datetime.now().astimezone().isoformat(),'expected_enabled':False,'ranks':rows};(d/'effective_progress_env.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
