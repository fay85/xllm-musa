"""Read-only progress snapshot; does not submit requests or change any role."""
from pathlib import Path
import sys,json,shlex
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
state=json.loads((b/'证据/matrix_status.json').read_text());previous=json.loads((b/'证据/latest_observation.json').read_text()) if (b/'证据/latest_observation.json').exists() else {};run=state.get('run') or previous.get('run','stage1_swiglu_d_v2');assert run.replace('_','').isalnum()
code=r'''from pathlib import Path
import json,sqlite3,datetime
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops');run=RUN;out={'at':datetime.datetime.now().astimezone().isoformat(),'run':run,'ranks':{},'client':None,'groups':[]}
for host in ('sj9','sj10'):
 p=root/'runs'/run/host/'status.json';out['ranks'][host]=json.loads(p.read_text()) if p.exists() else None
p=root/'runs'/run/'acceptance/status.json';out['client']=json.loads(p.read_text()) if p.exists() else None
for shape in (256,2048):
 for phase in ('warmup','measured'):
  p=root/'03_benchmarks'/run/f'input{shape}_output256'/phase
  for db in p.rglob('benchmark_data.db'):
   try:
    with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True,timeout=1) as c:
     rows=c.execute('SELECT success,prompt_tokens,completion_tokens,latency FROM result').fetchall()
    out['groups'].append({'shape':f'{shape}/256','phase':phase,'completed':len(rows),'success':sum(bool(r[0]) for r in rows),'actual_lengths':sorted(set((r[1],r[2]) for r in rows))})
   except sqlite3.OperationalError as e:out['groups'].append({'shape':shape,'phase':phase,'reading':str(e)})
print(json.dumps(out))
'''.replace('RUN',repr(run))
with sj_access.ssh_client('sj10') as c:r=sj_access.execute(c,'sj10','python3 -c '+shlex.quote(code),root=True,timeout=20)
assert r['exit_status']==0,r;d=json.loads(r['stdout']);d['matrix']=state;(b/'证据/latest_observation.json').write_text(json.dumps(d,indent=2))
summary={'at':d['at'],'run':d['run'],'client':d['client'],'groups':d['groups'],'rank_exits':{h:r['exit_codes'] for h,r in d['ranks'].items() if r and any(x is not None for x in r['exit_codes'])},'matrix_phase':state['phase']}
if '--compact' in sys.argv:
 summary['groups']=[f"{g['shape']} {g['phase']} done={g.get('completed','?')}/{5 if g['phase']=='measured' else 1}, success={g.get('success','?')}" for g in d['groups']]
 summary['client']=(d['client'] or {}).get('phase')
print(json.dumps(summary))
