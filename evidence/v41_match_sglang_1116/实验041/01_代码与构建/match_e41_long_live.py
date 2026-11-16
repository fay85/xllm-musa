from pathlib import Path
import json,datetime
r=Path('/expt/runs/p41_static_activation_proposal5_v1');p=Path('/proc/55294');v={'at':datetime.datetime.now().astimezone().isoformat(),'long_live':p.exists() and p.joinpath('stat').read_text().split()[21]=='523007747'}
for name in ['long_smoke/result.json','long_smoke/finished.json','long_smoke/error.txt','long_smoke_supervisor.log']:
 f=r/name
 if f.exists():v[name]=f.read_text()[-1700:]
f=r/'long_smoke/response_sse.jsonl';v['response_bytes']=f.stat().st_size if f.exists() else None
v['dlog_tail']=(r/'sj10/rank0.log').read_text(errors='replace')[-1800:]
print(json.dumps(v,ensure_ascii=False))
