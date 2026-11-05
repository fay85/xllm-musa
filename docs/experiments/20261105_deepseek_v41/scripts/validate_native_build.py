"""Run CPU contract tests and record the new native binary's exact provenance."""
from pathlib import Path
import hashlib,json,os,subprocess,time
root=Path('/work');repo=root/'repos/xllm-musa';out=root/'evidence/native_validation_v1'
out.mkdir(exist_ok=False)
env=dict(os.environ);env['LD_PRELOAD']='';env['MUSA_VISIBLE_DEVICES']='-1'
env['LD_LIBRARY_PATH']=':'.join(['/work/artifacts/bin',str(repo/'build/cmake.linux-x86_64-cpython-310/mooncake-common'),'/usr/local/lib/python3.10/dist-packages/torch/lib','/usr/local/lib/python3.10/dist-packages/torch_musa/lib','/usr/local/musa/lib','/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib','/opt/intel/oneapi/mkl/lib/intel64','/usr/local/openmpi/lib','/usr/local/lib','/usr/lib/x86_64-linux-gnu'])
records=[]
for name in ['folly_bthread_compat_test','xservice_client_lifecycle_test','native_provider_cache_groups_test','draft_model_config_test','deepseek_v41_cpp_template_test','mtp_prefix_cache_test','concurrent_block_manager_test']:
 binary=root/'artifacts/bin'/name
 assert binary.is_file(),str(binary)
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as stream:
  process=subprocess.Popen([str(binary),f'--gtest_output=xml:{out/name}.xml'],stdout=stream,stderr=subprocess.STDOUT,env=env)
  try:code=process.wait(timeout=180)
  except subprocess.TimeoutExpired:
   process.terminate()
   try:process.wait(timeout=30)
   except subprocess.TimeoutExpired:pass
   code='TIMEOUT_TERM_SENT'
 record={'name':name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'exit':code,'elapsed_seconds':time.monotonic()-start};records.append(record)
 (out/'results.json').write_text(json.dumps(records,indent=2));print(json.dumps(record),flush=True)
 if isinstance(code,str):raise RuntimeError('Inspect hung CPU test before proceeding')
binary=root/'artifacts/bin/xllm'
for name,cmd in [('ldd',['ldd',str(binary)]),('helpxml',[str(binary),'--helpxml']),('git_status',['git','-c','safe.directory=*','-C',str(repo),'status','--short']),('git_head',['git','-c','safe.directory=*','-C',str(repo),'rev-parse','HEAD'])]:
 result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=45)
 (out/(name+'.txt')).write_text(result.stdout+'\nSTDERR:\n'+result.stderr)
records.append({'name':'xllm_binary','sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'size':binary.stat().st_size})
(out/'results.json').write_text(json.dumps(records,indent=2))
raise SystemExit(0 if all(r.get('exit',0)==0 for r in records) else 1)
