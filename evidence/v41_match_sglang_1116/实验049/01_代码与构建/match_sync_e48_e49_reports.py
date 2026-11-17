from pathlib import Path
import hashlib,tarfile,json,sys,datetime
w=Path.cwd();o=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它';stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');archive=Path('/tmp')/('match_reports_'+stamp+'.tar.gz');files=[]
for sub in ['00_总控与恢复', '03_累计最佳', '02_实验记录/实验_045', '02_实验记录/实验_046', '02_实验记录/实验_047', '02_实验记录/实验_048', '02_实验记录/实验_049', '02_实验记录/实验_050', '01_参考依据/SG_投影与mHC融合']:
 for p in (o/sub).rglob('*'):
  if p.is_file() and p.suffix in ['.md','.json','.png','.txt','.py'] and not p.name.endswith(('.pt.trace.json','.native_stacks.jsonl')) and p.stat().st_size<2_000_000:files.append(p)
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
 for p in files:t.add(p,arcname=str(p.relative_to(o)))
sha=hashlib.sha256(archive.read_bytes()).hexdigest();remote='/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/inputs/'+archive.name
sys.path.insert(0,str(w/'sj9_12_登陆工具'));import sj_access as a
with a.ssh_client('sj9') as client:
 with client.open_sftp() as sftp:sftp.put(str(archive),'/tmp/'+archive.name)
 code="""from pathlib import Path
import json,hashlib,tarfile,shutil
p=Path(ARCHIVE);upload=Path('/tmp')/p.name;assert hashlib.sha256(upload.read_bytes()).hexdigest()==SHA;shutil.copy2(upload,p);upload.unlink();assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA;r=p.parent.parent;out=r/'reports';backup=r/'00_control/recovery_snapshots'/STAMP;backup.mkdir(exist_ok=False)
with tarfile.open(p) as t:
 members=t.getmembers();assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.isfile() for m in members)
 for m in members:
  target=out/m.name
  if target.exists():
   dest=backup/m.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest)
 t.extractall(out)
print(json.dumps({'files':len(members),'archive':str(p),'sha256':SHA,'previous_snapshot':str(backup)}))
""".replace('ARCHIVE',repr(remote)).replace('SHA',repr(sha)).replace('STAMP',repr(stamp))
 z=a.execute(client,'sj9','python3 -',root=True,timeout=60,payload=code)
(o/'00_总控与恢复'/('reports_sync_'+stamp+'.json')).write_text(json.dumps(z,ensure_ascii=False,indent=2));print(z['stdout']);assert z['exit_status']==0,z['stderr']
