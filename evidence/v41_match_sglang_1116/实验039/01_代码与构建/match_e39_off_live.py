from pathlib import Path
import json
r=Path('/expt/runs/p39_static_activation_off_v1');v={}
for n in ['formal_client_launch.json','formal_client_supervisor.log','sj9/supervisor.log','sj10/supervisor.log','sj9/rank0.log','sj10/rank0.log']:
 p=r/n
 if p.exists():v[n]=p.read_text()[-2800:]
for h in ['sj9','sj10']:v[h+'_role_exists']=(r/h/'role_config.json').exists()
j=json.loads((r/'formal_client_launch.json').read_text());p=Path('/proc',str(j['pid']));v['client_identity_live']=p.exists() and (p/'stat').read_text().split()[21]==j['start_ticks']
print(json.dumps(v))
