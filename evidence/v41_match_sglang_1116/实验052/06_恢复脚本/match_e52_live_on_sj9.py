from pathlib import Path
import json,datetime,os
r=Path('/expt/runs/p52_rope_table_on_v1')/'sj9'
v={'at':datetime.datetime.now().astimezone().isoformat(),'host':'sj9'}
p=r/'supervisor_job.json';job=json.loads(p.read_text());q=Path('/proc')/str(job['pid']);v['supervisor_live']=q.exists() and q.joinpath('stat').read_text().split()[21]==str(job['start_ticks'])
s=json.loads((r/'status.json').read_text());v['exit_codes']=s.get('exit_codes');v['ranks']=[]
for pid in s.get('pids',[]):
 p=Path('/proc')/str(pid)
 if not p.exists():v['ranks'].append({'pid':pid,'live':False});continue
 try:v['ranks'].append({'pid':pid,'start_ticks':p.joinpath('stat').read_text().split()[21],'comm':p.joinpath('comm').read_text().strip(),'exe':os.readlink(p/'exe')})
 except FileNotFoundError:v['ranks'].append({'pid':pid,'live':False})
p=r/'rank0.log';v['rank0_tail']=p.read_text(errors='replace')[-1800:] if p.exists() else ''
print(json.dumps(v))
