from pathlib import Path
import json
p=Path('/expt/diagnostics/e51_gather_gpu_v2_bits')
r={}
for n in ['status.json','finished.json','normal_cleanup.json']:
 f=p/n
 if f.exists():r[n]=json.loads(f.read_text())
r['ranks']=[]
for i in range(8):
 f=p/f'rank{i}.json';log=p/f'rank{i}.log'
 x=json.loads(f.read_text()) if f.exists() else {}
 r['ranks'].append({'rank':i,'status':x.get('status'),'stage':x.get('stage'),'checks':len(x.get('checks',[])),'microbenchmarks':len(x.get('microbenchmarks',[])),'log_tail':log.read_text(errors='replace')[-1000:] if log.exists() else None})
f=Path('/expt/diagnostics/e51_gather_supervisor_v2_bits.log')
r['supervisor_log']=f.read_text(errors='replace')[-1000:] if f.exists() else None
print(json.dumps(r))
