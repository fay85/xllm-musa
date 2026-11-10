"""Download a completed experiment and verify its archive before extraction."""
from pathlib import Path
import argparse,sys,shlex,json,hashlib,tarfile,datetime
b=Path(__file__).resolve().parents[1];sys.path.insert(0,str(b.parents[2]/'sj9_12_登陆工具'));import sj_access
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--bench');p.add_argument('--include-failed',action='store_true');p.add_argument('--tag',default='');a=p.parse_args();assert not a.tag or a.tag.replace('_','').isalnum();assert a.run.replace('_','').isalnum();bench=a.bench or a.run;archive_name=a.run+('_'+a.tag if a.tag else '')
assert not (b/'证据/完整实验'/archive_name).exists(), 'Use a new --tag; never overwrite a prior snapshot'
assert not (b/'证据'/f'{archive_name}_archive_receipt.json').exists(), 'A receipt already exists for this snapshot'
root='/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops'
code='''from pathlib import Path
import tarfile,hashlib,json
root=Path(ROOT);run=RUN;bench=BENCH;archive_name=ARCHIVE_NAME
status=json.loads((root/'03_benchmarks'/bench/'validated_summary.json').read_text())['status'];assert status=='PASS' or INCLUDE_FAILED
assert all((root/'runs'/run/h/'result.json').exists() for h in ('sj9','sj10'))
items=['03_benchmarks/'+bench,'runs/'+run,'artifacts/prepared/'+run]
items += ['02_correctness/'+run+x for x in ('','_prefix','_chat_prefix','_client_diagnostic','_startup503_attempt1') if (root/('02_correctness/'+run+x)).exists()]
out=root/'incoming'/('completed_'+archive_name+'.tar.gz')
with tarfile.open(out,'x:gz') as t:
 for item in items:t.add(root/item,arcname=item)
record={'archive_name':archive_name,'validation_status':status,'run':run,'benchmark':bench,'items':items,'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'path':str(out)}
print(json.dumps(record))
'''.replace('ROOT',repr(root)).replace('RUN',repr(a.run)).replace('BENCH',repr(bench)).replace('INCLUDE_FAILED',repr(a.include_failed)).replace('ARCHIVE_NAME',repr(archive_name))
with sj_access.ssh_client('sj10') as c:
 r=sj_access.execute(c,'sj10','python3 -c '+shlex.quote(code),root=True,timeout=45);assert r['exit_status']==0,r;d=json.loads(r['stdout'])
 archive=b/'传输缓存'/Path(d['path']).name
 with c.open_sftp() as f:f.get(d['path'],str(archive),prefetch=False)
assert archive.stat().st_size==d['bytes'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==d['sha256']
out=b/'证据/完整实验'/archive_name;out.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive) as t:
 for item in t.getmembers():
  target=(out/item.name).resolve();assert target.is_relative_to(out.resolve()) and not item.issym() and not item.islnk()
 t.extractall(out)
d['local_path']=str(out);d['verified_at']=datetime.datetime.now().astimezone().isoformat();archive.unlink();d['local_transfer_archive_removed']=True
(b/'证据'/f'{archive_name}_archive_receipt.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print(json.dumps(d,ensure_ascii=False))
