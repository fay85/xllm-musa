from pathlib import Path
import os,subprocess,json,datetime,hashlib,shlex
r=Path('/expt/builds/e52_rope_table_module_v2');s=Path('/workspace/xllm-mt-main');b=s/'build/cmake.linux-x86_64-cpython-310';ref=Path('/expt/diagnostics/e52_rope_build_inputs');old=Path('/expt/builds/e50_native_gate_gemv_v3')
prev=json.loads((ref/'e50_native_gate_gemv_v3_commands.json').read_text());cpp=prev[0];cpp[cpp.index('-c')+1]=str(r/'src/v41_rope_table_test_module.cpp');cpp[cpp.index('-o')+1]=str(r/'module.o')
cu=shlex.split(json.loads((ref/'rope_compile_command.json').read_text())['command']);cu.insert(1,'-DC10_USE_GLOG');cu[cu.index('-c')+1]=str(r/'src/v41_rope_table_candidate.cu');cu[cu.index('-o')+1]=str(r/'rope.o')
link=[x.replace(str(old/'v41_gate_gemv_test_module.so'),str(r/'v41_rope_table_test_module.so')).replace(str(old/'0.o'),str(r/'module.o')) for x in prev[1]];link.insert(link.index(str(r/'module.o'))+1,str(r/'rope.o'))
env=dict(os.environ,LD_PRELOAD='',MUSA_VISIBLE_DEVICES='-1',PYTHONPATH=str(r),LD_LIBRARY_PATH='/expt/artifacts/native_sg438_control_v1/lib/runtime:/opt/venv/lib/python3.10/site-packages/torch_musa/lib:/opt/venv/lib/python3.10/site-packages/torch/lib:/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib:/usr/local/musa/lib',PATH=str(b/'musa_link_wrapper')+':/opt/venv/bin:/usr/local/musa/bin:/usr/bin:/bin',MUSAMAPPING_PATH='/usr/local/musa/tools/musamapping',TVM_FFI_LIB_DIR='/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib',LIBRARY_PATH=str(b/'xllm/core/layers/musa')+':/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64')
(r/'commands.json').write_text(json.dumps([cpp,cu,link],indent=2));rc=1
try:
 for name,cmd in [('cpp',cpp),('cu',cu),('link',link)]:
  with (r/(name+'.log')).open('wb') as f:p=subprocess.run(cmd,cwd=b,env=env,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,name+' failed'
 p=subprocess.run(['/opt/venv/bin/python3','-c','import torch,torch_musa;import v41_rope_table_test_module as m;assert hasattr(m,"candidate");print("IMPORT_PASS")'],env=env,capture_output=True,text=True);(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'import failed';rc=0
finally:
 v={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()};p=r/'v41_rope_table_test_module.so'
 if p.exists():v.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 (r/'finished.json').write_text(json.dumps(v))
