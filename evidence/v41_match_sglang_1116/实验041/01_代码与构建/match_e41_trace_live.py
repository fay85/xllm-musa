from pathlib import Path
import json,datetime
r=Path('/expt/runs/p41_static_activation_proposal5_v1');p=Path('/proc/55273');v={'at':datetime.datetime.now().astimezone().isoformat(),'trace_live':p.exists() and p.joinpath('stat').read_text().split()[21]=='523002182'}
for n in ['trace/requests.json','trace/finished.json','trace/error.txt','trace_supervisor.log','trace/start_profile.json','trace/stop_profile.json']:
 f=r/n
 if f.exists():v[n]=f.read_text()[-2500:]
print(json.dumps(v,ensure_ascii=False))
