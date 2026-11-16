from pathlib import Path
import json
r=Path('/expt/runs/p39_static_activation_off_v1');print('CLIENT_SOURCE_HEAD','\n'.join(Path('/expt/scripts/match_e39_formal_client.py').read_text().splitlines()[:28]));print('TOP',[x.name for x in r.iterdir()])
for p in r.rglob('active.json'):print(str(p),p.read_text()[:2000])
for p in r.rglob('summary.json'):print(str(p),p.read_text()[:2000])
for p in r.rglob('error.txt'):print(str(p),p.read_text()[-1000:])
print('PDSUMMARIES')
for h in ['sj9','sj10']:
 lines=(r/h/'rank0.log').read_text().splitlines();print(h, '\n'.join(x for x in lines if any(y in x for y in ['generation speed','Graph','graph','DSpark summary']))[-4500:])
