from pathlib import Path
import json,subprocess,datetime,socket,time
h='sj9' if socket.gethostname()=='worker32007' else 'sj10';names={'sj9':('xiaopang1116_v41_match_prefill_sj9','68b4f202f5821bd9b8d819b2b564609145de5f141a7f4c10256bc0e5cba1c984'),'sj10':('xiaopang1116_v41_match_decode_sj10','d7a1e84d42b8c8bf1e7e51444ccc465fd81e529c047fa49142790c8921d52a63')};name,cid=names[h];c=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert c['Id']==cid and c['State']['Running'];r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116')
for host in ['sj9','sj10']:
 p=json.loads((r/'00_control'/f'{host}_preflight_p0.json').read_text());assert p['gpu_free'];assert time.time()-datetime.datetime.fromisoformat(p['at']).timestamp()<180,'Stale resource preflight'
g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True,timeout=25);assert g.returncode==0 and 'No running processes found' in g.stdout,'GPU changed since preflight'
ports=subprocess.check_output(['ss','-ltnH'],text=True);used={int(x.split()[3].rsplit(':',1)[-1]) for x in ports.splitlines()};needed={22000,22100,22200,22300} if h=='sj9' else {32000,32100,32200,32300,31178,31179,31180,31181,31182,31183};assert not used&needed,used&needed
payload="""from pathlib import Path
import subprocess,json,os,datetime
r=Path('/expt/runs/p3_attention_replica_graph_v1')/HOST
assert not (r/'supervisor_job.json').exists() and not (r/'rank0.log').exists()
with (r/'supervisor.log').open('xb') as f:
 p=subprocess.Popen(['python3','/expt/scripts/pd_rank_supervisor_nocore_v1.py','--config',str(r/'prepared_config.json'),'--output',str(r)],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc') .joinpath(str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'host':HOST};(r/'supervisor_job.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
""".replace('HOST',repr(h))
p=subprocess.run(['docker','exec','-i',name,'python3','-'],input=payload,text=True,capture_output=True,timeout=30);print(p.stdout);print(p.stderr);assert p.returncode==0
