from pathlib import Path
import argparse,hashlib,json,sys,tarfile
ap=argparse.ArgumentParser();ap.add_argument('--receipt',type=Path,required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
assert a.name in ['final_short1_v1','final_short2_v1','final_short3_v1','final_long_v1','sglang_B_recheck_v1']
v=json.loads(a.receipt.read_text());v=json.loads(v['stdout'].strip().splitlines()[-1])
w=Path.cwd();sys.path.insert(0,str(w/'sj9_12_登陆工具'));import sj_access as access
out=Path(__file__).parent.parent/'原始验收证据';out.mkdir(exist_ok=True);target=out/(a.name+'.tar.gz');assert not target.exists()
with access.ssh_client('sj9') as client:
 with client.open_sftp() as sftp:sftp.get(v['archive'],str(target))
assert target.stat().st_size==v['bytes'] and hashlib.sha256(target.read_bytes()).hexdigest()==v['sha256']
dest=out/a.name;dest.mkdir(exist_ok=False)
with tarfile.open(target) as tar:
 assert all(not Path(x.name).is_absolute() and '..' not in Path(x.name).parts and not x.issym() and not x.islnk() for x in tar.getmembers())
 tar.extractall(dest)
(out/(a.name+'_receipt.json')).write_text(json.dumps(v,indent=2))
print(json.dumps({'name':a.name,'bytes':v['bytes'],'sha256':v['sha256'],'local':str(dest)}))
