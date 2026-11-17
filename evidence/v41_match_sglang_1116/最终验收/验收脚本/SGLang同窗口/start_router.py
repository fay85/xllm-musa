"""Host SJ9 payload: start only our frozen B CPU router after own role launches."""
from pathlib import Path
import datetime,json,socket,subprocess
assert socket.gethostname()=='worker32007'
name='xiaopang1116_sglang_prefill_sj9'
c=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
assert c['Id']=='034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2' and c['State']['Running']
root=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117')
assert all((root/'runtime'/role/'launcher.json').is_file() for role in ['prefill','decode'])
ports={int(x.split()[3].rsplit(':',1)[-1]) for x in subprocess.check_output(['ss','-ltnH'],text=True).splitlines()}
assert not ports&{28192,29031}
payload=r"""
from pathlib import Path
import datetime,json,os,subprocess
r=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117')
with (r/'runtime/router.log').open('xb') as log:
 p=subprocess.Popen(['bash',str(r/'configs/router.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,MUSA_VISIBLE_DEVICES='-1'))
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()}
(r/'runtime/router_job.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
"""
z=subprocess.run(['docker','exec','-i',name,'/opt/venv/bin/python','-'],input=payload,text=True,capture_output=True,timeout=30)
print(z.stdout);print(z.stderr);assert z.returncode==0

