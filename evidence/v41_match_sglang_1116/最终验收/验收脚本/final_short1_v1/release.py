"""Host payload: record own release even when an external job has since started."""
from pathlib import Path
import datetime,hashlib,json,socket,subprocess,tarfile
host={'worker32007':'sj9','worker32009':'sj10'}[socket.gethostname()]
root=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116')
run=root/'runs'/"final_short1_v1"
name,cid={'sj9':('xiaopang1117_v41_sg438_prefill_sj9','52abeddab774ff06522e2ffe452b0714065c579f95137660f51f54ed3d9ec94a'),'sj10':('xiaopang1117_v41_sg438_decode_sj10','7eefcf3aedcb80825833e6b47295689965905518e51823cab8bedecfad6374f3')}[host]
c=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert c['Id']==cid
done=json.loads((run/host/'result.json').read_text());assert done['exit_codes']==[0]*8 and not done.get('automatic_kill',False)
top=subprocess.check_output(['docker','top',name,'-eo','pid,comm'],text=True)
assert not any(line.split()[-1] in ['xllm','xllm-server'] for line in top.splitlines()[1:])
g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True,timeout=25)
v={'at':datetime.datetime.now().astimezone().isoformat(),'host':host,'own_container':name,'rank_exit_codes':done['exit_codes'],'own_resources_released':True,'gpu_free':g.returncode==0 and 'No running processes found' in g.stdout,'gpu_stdout':g.stdout,'gpu_stderr':g.stderr,'own_container_pid_comm':top}
(run/host/'resource_release.json').write_text(json.dumps(v,indent=2))
print(json.dumps({k:v[k] for k in ['at','host','own_resources_released','gpu_free']}))

