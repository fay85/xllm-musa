from pathlib import Path
import json,tarfile,hashlib,datetime
r=Path('/expt');p=r/'diagnostics/e40_prequant_real_weights_v1';assert json.loads((p/'finished.json').read_text())['exit_code']==0
v=json.loads((p/'result.json').read_text());assert v['status']=='PASS' and len(v['checks'])==72 and len(v['benchmarks'])==12
out=r/'exports/e40_prequant_real_weights_v1.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for n in ['probe.py','probe.log','worker.py','worker.log','result.json','partial.json','finished.json','job.json','child.json']:t.add(p/n,arcname='probe/'+n)
 for n in ['fp8_capacity_delta.json','parameter_budget.json']:t.add(r/'diagnostics/e40_dp8_memory_budget'/n,arcname='budget/'+n)
z={'archive':str(out),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'checks':len(v['checks']),'max_RMS':max(x['rms'] for x in v['checks']),'benchmarks':[{k:x[k] for k in ['name','rows','weight_shape']}|{'control_us':x['control']['median_us'],'prequant_us':x['prequant']['median_us'],'reverse_us':x['control_reverse']['median_us']} for x in v['benchmarks']]};print(json.dumps(z))
