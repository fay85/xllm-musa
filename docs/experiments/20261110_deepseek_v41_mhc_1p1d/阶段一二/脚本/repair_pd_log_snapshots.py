from pathlib import Path
import hashlib,json,datetime
root=Path('/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/runs/stage1_ref');p=root/'pd_evidence.json';old=p.read_bytes();d=json.loads(old);(root/'pd_evidence_before_snapshot_repair.json').write_bytes(old);out=root/'pd_evidence_logs';out.mkdir(exist_ok=False);rows=[]
for r in d['rows']:
 source=root/r['host']/f"rank{r['rank']}.log";raw=source.read_bytes();h=hashlib.sha256();offset=0;found=None
 for line in raw.splitlines(keepends=True):
  h.update(line);offset+=len(line)
  if h.hexdigest()==r['log_sha256']:found=offset;break
 assert found is not None,source
 target=out/f"{r['host']}_rank{r['rank']}.log";target.write_bytes(raw[:found]);assert hashlib.sha256(target.read_bytes()).hexdigest()==r['log_sha256']
 r.update(log_snapshot=str(target.relative_to(root)),source_live_path=str(source));rows.append({'host':r['host'],'rank':r['rank'],'sha256':r['log_sha256'],'snapshot_bytes':found})
p.write_text(json.dumps(d,indent=2));record={'at':datetime.datetime.now().astimezone().isoformat(),'status':'EXACT_SIXTEEN_HASHED_PREFIXES_RECOVERED','rows':rows,'reason':'Normal shutdown appended to live logs after PD transfer validation. Preserve exactly the byte prefixes corresponding to the previously recorded hashes.'};(root/'pd_log_snapshot_repair.json').write_text(json.dumps(record,indent=2));print(json.dumps({'status':record['status'],'ranks':len(rows)}))
