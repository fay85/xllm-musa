from pathlib import Path
import sys,json,base64
w=Path.cwd();src=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/03_代码/xllm-musa/tools/musa_deepseek_v41/validate_native_sparse_mate.py';data=base64.b64encode(src.read_bytes()).decode()
code='''from pathlib import Path
import subprocess,json,base64,datetime,os
name='xiaopang1116_v41_match_prefill_sj9';cid='68b4f202f5821bd9b8d819b2b564609145de5f141a7f4c10256bc0e5cba1c984';c=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert c['Id']==cid and c['State']['Running'];g=subprocess.run(['timeout','20','mthreads-gmi'],capture_output=True,text=True);assert g.returncode==0 and 'No running processes found' in g.stdout
r=Path('/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/builds/attention_replica_v1/small_checks_v7');(r/'preflight_gmi.txt').write_text(g.stdout);(r/'validate_native_sparse_mate.py').write_bytes(base64.b64decode(DATA))
payload=PAYLOAD
p=subprocess.run(['docker','exec','-i',name,'python3','-'],input=payload,text=True,capture_output=True,timeout=30);print(p.stdout,p.stderr);assert p.returncode==0
'''
payload='''from pathlib import Path
import json,subprocess,os,datetime
r=Path('/expt/builds/attention_replica_v1/small_checks_v7');assert json.loads((r/'finished.json').read_text())['exit_code']==0;assert not (r/'gpu_job.json').exists()
worker=WORKER
(r/'gpu_worker.py').write_text(worker)
with (r/'gpu_worker.log').open('xb') as f:p=subprocess.Popen(['python3',str(r/'gpu_worker.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21]}))
'''
worker='''from pathlib import Path
import json,subprocess,os,datetime,hashlib
r=Path('/expt/builds/attention_replica_v1/small_checks_v7');cfg=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj9/role_config.json').read_text());base=cfg['ranks'][0]['environment'];env=dict(os.environ,LD_LIBRARY_PATH=base['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r)+':/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XLLM_V41_LIGHT_PREFIX='')
(r/'gpu_job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'gpu':'SJ9 GPU0'},indent=2));tests=[('sparse_h64',['python3',str(r/'validate_native_sparse_mate.py'),'--module','v41_attention_layout_test_module','--heads','64','--graph','--library','/expt/builds/attention_replica_v1/sparse_h64/artifact/libdeepseek_v41_sparse_h64_bf16.so','--output',str(r/'sparse_h64_checks.json')]),('projection',['python3',str(r/'validate_attention_replication.py'),'--output',str(r/'projection_checks.json')])];records=[];rc=1
try:
 for name,cmd in tests:
  with (r/(name+'.log')).open('xb') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  active={'name':name,'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'command':cmd};(r/'gpu_active.json').write_text(json.dumps(active,indent=2));p.wait();records.append(dict(active,exit_code=p.returncode));(r/'gpu_status.json').write_text(json.dumps(records,indent=2));assert p.returncode==0,'GPU check failed: '+name
 rc=0
finally:
 (r/'gpu_finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
'''
payload=payload.replace('WORKER',repr(worker));code=code.replace('DATA',repr(data)).replace('PAYLOAD',repr(payload));sys.path.insert(0,str(w/'sj9_12_登陆工具'));import sj_access as a
with a.ssh_client('sj9') as c:v=a.execute(c,'sj9','python3 -',root=True,timeout=50,payload=code)
print(v['stdout'],v['stderr']);assert v['exit_status']==0
