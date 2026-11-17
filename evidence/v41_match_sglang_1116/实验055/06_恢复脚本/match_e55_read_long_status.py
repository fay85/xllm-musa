from pathlib import Path
import json
r=Path('/expt/runs/p55_draft_gate_on_v1')
out={}
for n in ['long_smoke/finished.json','long_smoke/result.json','long_smoke/error.txt','long_smoke_supervisor.log']:
 p=r/n
 if p.exists():out[n]=p.read_text()[-4000:]
p=r/'long_smoke/response_sse.jsonl'
if p.exists():out['sse_bytes']=p.stat().st_size;out['sse_tail']=p.read_text()[-1400:]
out['d_log_tail']=(r/'sj10/rank0.log').read_text(errors='replace')[-2200:]
print(json.dumps(out,ensure_ascii=False))
