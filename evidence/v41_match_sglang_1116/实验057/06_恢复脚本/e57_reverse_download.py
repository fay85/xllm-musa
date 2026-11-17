from pathlib import Path
import sys,json,hashlib,tarfile
w=Path.cwd();sys.path.insert(0,str(w/'sj9_12_登陆工具'));import sj_access as a
s=Path(__file__).parent;e=s.parent
receipt=json.loads((s/'E57反向归档.json').read_text());assert receipt['exit_status']==0
v=json.loads(receipt['stdout'].strip().splitlines()[-1]);assert v['archive']=='/expt/exports/p57_progress_reverse_enabled_v1_evidence.tar.gz'
p=e/'原始证据_reverse.tar.gz';assert not p.exists()
with a.ssh_client('sj9') as c:
 with c.open_sftp() as f:f.get(v['archive'].replace('/expt','/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116',1),str(p))
assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];out=e/'原始证据_reverse';out.mkdir(exist_ok=False)
with tarfile.open(p) as t:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers());t.extractall(out)
(e/'归档回执_reverse.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
