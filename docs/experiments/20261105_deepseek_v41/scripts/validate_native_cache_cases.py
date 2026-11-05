"""Isolate every cache case so a baseline assertion cannot hide later V4.1 cases."""
from pathlib import Path
import concurrent.futures,json,os,resource,subprocess,time
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path('/work');out=root/'evidence/native_cache_cases_v1';out.mkdir(exist_ok=False)
binary=root/'artifacts/bin/mtp_prefix_cache_test'
env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1')
env['LD_LIBRARY_PATH']=':'.join(['/work/artifacts/bin','/work/repos/xllm-musa/build/cmake.linux-x86_64-cpython-310/mooncake-common','/usr/local/lib/python3.10/dist-packages/torch/lib','/usr/local/lib/python3.10/dist-packages/torch_musa/lib','/usr/local/musa/lib','/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib','/opt/intel/oneapi/mkl/lib/intel64','/usr/local/openmpi/lib','/usr/local/lib','/usr/lib/x86_64-linux-gnu'])
r=subprocess.run([str(binary),'--gtest_list_tests'],env=env,capture_output=True,text=True,timeout=30);r.check_returncode();(out/'test_list.txt').write_text(r.stdout)
suite='';cases=[]
for line in r.stdout.splitlines():
 line=line.split('#')[0].rstrip()
 if line and not line.startswith(' ') and line.endswith('.'):suite=line
 elif line.startswith('  ') and suite:cases.append(suite+line.strip())
assert len(cases)==264,len(cases)
def run(name):
 start=time.monotonic();safe=name.replace('/','_');log=out/(safe+'.log')
 with log.open('w') as stream:
  p=subprocess.Popen([str(binary),'--gtest_filter='+name],env=env,stdout=stream,stderr=subprocess.STDOUT)
  try:code=p.wait(timeout=60)
  except subprocess.TimeoutExpired:
   p.terminate()
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:pass
   code='TIMEOUT_TERM_SENT'
 return {'name':name,'exit':code,'elapsed_s':time.monotonic()-start,'log':log.name}
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for row in pool.map(run,cases):
  results.append(row);(out/'results.json').write_text(json.dumps(results,indent=2))
  if row['exit']!=0:print(json.dumps(row),flush=True)
summary={'total':len(results),'passed':sum(x['exit']==0 for x in results),'failed':[x for x in results if x['exit']!=0],'v41':[x for x in results if 'V41' in x['name']]}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['failed','v41']}))
