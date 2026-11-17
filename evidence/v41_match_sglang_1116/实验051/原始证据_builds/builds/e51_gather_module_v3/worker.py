from pathlib import Path
import os,subprocess,json,datetime,hashlib
r=Path('/expt/builds/e51_gather_module_v3');old=Path('/expt/builds/e44_mixed_mccl_module_v2');s=Path('/workspace/xllm-mt-main');b=s/'build/cmake.linux-x86_64-cpython-310';commands=json.loads((old/'commands.json').read_text());cpp=commands[0];cpp[cpp.index('-c')+1]=str(r/'src/tests/core/kernels/musa/v41_gather_candidate_test_module.cpp');cpp[cpp.index('-o')+1]=str(r/'module.o');cpp.insert(1,'-I'+str(s/'xllm/core'));cpp.insert(1,'-DMARCH_TYPE=310');cpp.insert(1,'-I'+str(r/'src/xllm'));link=[x.replace(str(old/'module.o'),str(r/'module.o')).replace(str(old/'v41_deep_ep_w4a8_test_module.so'),str(r/'v41_gather_candidate_test_module.so')) for x in commands[1]]
link=[x for x in link if not x.endswith('/e38_deepep_w4a8_v2/w4a8.o')]
link[link.index('-Wl,--start-group')+1:link.index('-Wl,--start-group')+1]=[str(b/'xllm/core/framework/parallel_state/libparallel_state.a')]
env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1',PYTHONPATH=str(r),LD_LIBRARY_PATH='/expt/artifacts/native_sg438_control_v1/lib/runtime:/opt/venv/lib/python3.10/site-packages/torch_musa/lib:/opt/venv/lib/python3.10/site-packages/torch/lib:/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib:/usr/local/musa/lib',PATH=str(b/'musa_link_wrapper')+':/opt/venv/bin:/usr/local/musa/bin:/usr/bin:/bin',TVM_FFI_LIB_DIR='/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib',LIBRARY_PATH=str(b/'xllm/core/layers/musa')+':/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64')
(r/'commands.json').write_text(json.dumps([cpp,link],indent=2));rc=1
try:
 for name,cmd in [('cpp',cpp),('link',link)]:
  with (r/(name+'.log')).open('wb') as f:p=subprocess.run(cmd,cwd=b,env=env,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,name+' failed'
 p=subprocess.run(['/opt/venv/bin/python3','-c','import torch,torch_musa;import v41_gather_candidate_test_module as m;assert hasattr(m,"GatherCandidate");print("IMPORT_PASS")'],env=env,capture_output=True,text=True);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'import failed';rc=0
finally:
 v={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()};p=r/'v41_gather_candidate_test_module.so'
 if p.exists():v.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 (r/'finished.json').write_text(json.dumps(v))
