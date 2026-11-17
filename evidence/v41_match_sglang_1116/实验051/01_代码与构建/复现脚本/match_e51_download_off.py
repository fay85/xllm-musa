from pathlib import Path
import sys,json,hashlib,tarfile
w=Path.cwd();sys.path.insert(0,str(w/'sj9_12_登陆工具'));import sj_access as a
mode='off_v1';z=json.loads(Path('/tmp/match_e51_off_archive_receipt.json').read_text());v=json.loads(z['stdout'].strip().splitlines()[-1]);p=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它/02_实验记录/实验_051';p.mkdir(parents=True,exist_ok=True);target=p/('原始证据_'+mode+'.tar.gz')
with a.ssh_client('sj9') as client:
 with client.open_sftp() as sftp:sftp.get(v['archive'].replace('/expt','/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116',1),str(target))
assert hashlib.sha256(target.read_bytes()).hexdigest()==v['sha256'];out=p/('原始证据_'+mode);out.mkdir(exist_ok=True)
with tarfile.open(target) as t:
 assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and not m.issym() and not m.islnk() for m in t.getmembers());t.extractall(out)
(p/('归档回执_'+mode+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print({'mode':mode,'bytes':target.stat().st_size,'sha256':v['sha256']})
