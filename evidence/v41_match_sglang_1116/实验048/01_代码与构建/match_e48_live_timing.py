from pathlib import Path
import json,collections,statistics,datetime
r=Path('/expt/runs/p48_runtime_boundaries_v1');out={'at':datetime.datetime.now().astimezone().isoformat()};d=r/'formal_client'
for n in ['active.json','summary.json','finished.json']:
 p=d/n
 if p.exists():out[n]=json.loads(p.read_text())
f=sorted((r/'runtime_timing').glob('phase.*.jsonl'));out['timing_files']=len(f);out['sample_counts']={p.name:sum(1 for x in p.open()) for p in f}
if f:
 lines=f[0].read_text().splitlines();vals=[json.loads(x) for x in lines];phases={}
 for v in vals:
  for b in v['boundaries']:
   if b.get('gpu_stream_span_ms') is not None:phases.setdefault(b['name'],[]).append(b['gpu_stream_span_ms'])
 out['first_file_phase_median_ms']={k:statistics.median(v) for k,v in phases.items()};out['first_sample']=vals[0] if vals else None
print(json.dumps(out))
