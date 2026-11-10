"""Recover immutable copies of the exact log prefixes hashed at validation."""
from pathlib import Path
import hashlib,json,datetime
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops')
b=root/'03_benchmarks/stage1_ref_v2';summary=b/'validated_summary.json';old=summary.read_bytes();d=json.loads(old);snap=b/'server_logs_at_validation';snap.mkdir(exist_ok=False)
(b/'validated_summary_before_log_snapshot_repair.json').write_bytes(old)
rows=[]
for row in d['server_log_provenance']:
 live=root/'runs/stage1_ref'/row['path'];raw=live.read_bytes();h=hashlib.sha256();offset=0;match=None
 for line in raw.splitlines(keepends=True):
  h.update(line);offset+=len(line)
  if h.hexdigest()==row['sha256']:match=offset;break
 assert match is not None,f'Cannot find exact hashed prefix for {live}'
 target=snap/(row['path'].replace('/','_'));target.write_bytes(raw[:match]);assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
 rows.append({'snapshot_path':str(target.relative_to(b)),'source_live_path':str(live),'sha256':row['sha256'],'snapshot_bytes':match,'source_bytes_at_recovery':len(raw)})
d['server_log_provenance']=rows
summary.write_text(json.dumps(d,ensure_ascii=False,indent=2))
record={'at':datetime.datetime.now().astimezone().isoformat(),'reason':'The initial validator hashed live logs. Subsequent diagnostic requests and normal shutdown appended to them. Exact original byte prefixes have been recovered and verified against the previously recorded SHA256 values; metrics and raw requests are unchanged.','status':'EXACT_HASHED_PREFIXES_RECOVERED','logs':rows}
(b/'log_snapshot_repair.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
