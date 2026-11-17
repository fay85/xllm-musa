from pathlib import Path
import json,datetime
r=Path('/expt/runs/p55_draft_gate_on_v1');a=json.loads((r/'formal_client/compilation_window.json').read_text());rows={}
for h,v in a['hosts'].items():
 rows[h]={'clear':v['clear'],'coverage':v['coverage'],'first_sample':v['observations'][0]['at'] if v['observations'] else None,'last_sample':v['observations'][-1]['at'] if v['observations'] else None,'busy':[{'at':x['at'],'processes':[{'pid':z['pid'],'comm':z['comm'],'containers':z['containers']} for z in x['build_processes']]} for x in v['observations'] if not x['compilation_clear']]}
print(json.dumps({'at':datetime.datetime.now().astimezone().isoformat(),'formal_window':a['formal_window'],'audit':rows}))
