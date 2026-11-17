from pathlib import Path
import json,datetime
root=Path('/data/nfs_shared/xiaopang');s=root/'sglang_v41_reference_20261116/B_recheck_20261117';r=root/'v41_xllm_match_sglang_20261116/runs/sglang_B_recheck_v1';v={'at':datetime.datetime.now().astimezone().isoformat()}
for role in ['prefill','decode','router']:
 d=s/'runtime'/role;v[role]={}
 for name in ['launch_result.json','launcher.json','stop_result.json']:
  if (d/name).exists():v[role][name]=json.loads((d/name).read_text())
 p=d/'server.log'
 if p.exists():v[role]['tail']=p.read_text(errors='replace')[-1800:]
p=s/'runtime/router.log'
if p.exists():v['router']['tail']=p.read_text(errors='replace')[-1800:]
for name in ['finished.json','performance/summary.json','client.log']:
 p=r/name
 if p.exists():v[name]=p.read_text(errors='replace')[-3000:]
print(json.dumps(v,ensure_ascii=False))
