from pathlib import Path
import subprocess,json,datetime
host='sj9';role='prefill';cid='034c599262ae8136bc51533e86ec79a2bccb27ef843e7f18269c430002bf00f2';name='xiaopang1116_sglang_'+role+'_'+host
r=Path('/data/nfs_shared/xiaopang/sglang_v41_reference_20261116/B_recheck_20261117/runtime')/role
c=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert c['Id']==cid and c['Config']['Labels']['owner']=='xiaopang' and c['Config']['Labels']['task']=='sglang-ab-trace-20261116'
assert (r/'stop_result.json').exists()
top=subprocess.check_output(['docker','top',name,'-eo','pid,comm,args'],text=True) if c['State']['Running'] else ''
assert not any('sglang::' in x or '-m sglang' in x for x in top.splitlines()[1:]),top
if c['State']['Running']:subprocess.run(['docker','stop','--time','20',name],check=True,capture_output=True,timeout=30)
after=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert not after['State']['Running']
g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True,timeout=25)
v={'at':datetime.datetime.now().astimezone().isoformat(),'host':host,'own_resources_released':True,'container_stopped':name,'container_id':cid,'gpu_free':g.returncode==0 and 'No running processes found' in g.stdout,'gpu_stdout':g.stdout,'gpu_stderr':g.stderr,'top_after_model_exit':top}
(r/'resource_release.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v[k] for k in ['host','own_resources_released','container_stopped','gpu_free']}))
