from pathlib import Path
import json,re,datetime
r=Path('/expt/runs');rows=[]
for name in ['p39_static_activation_off_v1','p39_static_activation_on_v1']:
 p=r/name/'sj10/rank0.log';lines=p.read_text(errors='replace').splitlines();selected=[x for x in lines if any(k in x.lower() for k in ['accept','speculat','generated','tpot','sequence finished','finish request','elapsed time'])];rows.append({'run':name,'selected_lines':selected[-80:]})
print(json.dumps(rows))
