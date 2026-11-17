from pathlib import Path
import time,json,datetime
r=Path('/expt/runs/p56_short_indexer_off_v1');deadline=time.monotonic()+45
while time.monotonic()<deadline and not all((r/h/'result.json').exists() for h in ['sj9','sj10']):time.sleep(2)
v={'at':datetime.datetime.now().astimezone().isoformat(),'both_finished':all((r/h/'result.json').exists() for h in ['sj9','sj10'])}
for h in ['sj9','sj10']:
 p=r/h/'result.json'
 v[h]=json.loads(p.read_text()) if p.exists() else json.loads((r/h/'status.json').read_text())
print(json.dumps(v))
