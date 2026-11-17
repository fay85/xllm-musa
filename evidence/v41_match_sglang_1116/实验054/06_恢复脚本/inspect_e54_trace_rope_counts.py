from pathlib import Path
import json,collections
r=Path('/expt/profiling/p54_rope_proposal_on_v1/sj10')
for f in sorted(r.glob('*.pt.trace.json')):
 d=json.loads(f.read_text());e=d['traceEvents'];g=[x for x in e if x.get('ph')=='X' and x.get('name')=='musaGraphLaunch'];kernels=[x for x in e if x.get('ph')=='X' and 'kernel' in x.get('cat','').lower()]
 rows=[]
 for x in g:
  nodes=[y for y in kernels if y.get('args',{}).get('correlation')==x['args']['correlation']]
  counts=collections.Counter(y['name'] for y in nodes if 'rope' in y['name'].lower())
  rows.append({'kernels':len(nodes),'rope':dict(counts)})
 print(json.dumps({'file':f.name,'graphs':rows}))
 break

