from pathlib import Path
import json,subprocess,os,datetime,hashlib
r=Path('/expt/builds/attention_replica_v1/small_checks');cfg=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj9/role_config.json').read_text());base=cfg['ranks'][0]['environment'];env=dict(os.environ,LD_LIBRARY_PATH=base['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(r)+':/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',XLLM_V41_LIGHT_PREFIX='')
(r/'gpu_job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'gpu':'SJ9 GPU0'},indent=2));tests=[('sparse_h64',['python3',str(r/'validate_native_sparse_mate.py'),'--module','v41_attention_layout_test_module','--heads','64','--graph','--library','/expt/builds/attention_replica_v1/sparse_h64/artifact/libdeepseek_v41_sparse_h64_bf16.so','--output',str(r/'sparse_h64_checks.json')]),('projection',['python3',str(r/'validate_attention_replication.py'),'--output',str(r/'projection_checks.json')])];records=[];rc=1
try:
 for name,cmd in tests:
  with (r/(name+'.log')).open('xb') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  active={'name':name,'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'command':cmd};(r/'gpu_active.json').write_text(json.dumps(active,indent=2));p.wait();records.append(dict(active,exit_code=p.returncode));(r/'gpu_status.json').write_text(json.dumps(records,indent=2));assert p.returncode==0,'GPU check failed: '+name
 rc=0
finally:
 (r/'gpu_finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
