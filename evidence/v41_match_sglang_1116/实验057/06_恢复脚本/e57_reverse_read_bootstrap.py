from pathlib import Path
import json
r=Path('/expt/runs/p57_progress_reverse_enabled_v1');v={}
for h in ['sj9','sj10']:
 d=r/h;v[h]={}
 for n in ['supervisor_job.json','supervisor.log','status.json','result.json','role_config.json','effective_progress_env.json']:
  p=d/n
  if p.exists():v[h][n]=p.read_text(errors='replace')[-1800:] if n!='role_config.json' else 'exists'
print(json.dumps(v))
