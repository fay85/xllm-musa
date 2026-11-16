from pathlib import Path
import json
r=Path('/expt')
for mode in ['off','on']:
 for h in ['sj9','sj10']:
  c=json.loads((r/f'runs/p39_static_activation_{mode}_v1/{h}/prepared_config.json').read_text());print(mode,h,'KEYS',list(c));print('RANKKEYS',list(c['ranks'][0]));print('OTHER',{k:v for k,v in c.items() if k not in ['ranks','treatment']})
for name in ['pd_rank_supervisor_nocore_v1.py','match_e39_formal_client.py']:
 s=(r/'scripts'/name).read_text();print('SCRIPT',name)
 for i,line in enumerate(s.splitlines()):
  if any(x in line for x in ['sha256','source','artifact','lib','stop','terminate','finish','sys.exit']):print(i+1,line[:250])
