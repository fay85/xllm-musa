from pathlib import Path
import base64
repo=Path('王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/03_代码/xllm-musa');names=['xllm/core/kernels/musa/deepseek_v41_dense_block32.cpp','xllm/core/kernels/musa/deepseek_v41_dense_block32.h','xllm/core/kernels/musa/musa_aot_kernel.cpp','xllm/core/kernels/musa/musa_aot_kernel.h','tests/core/kernels/musa/v41_prequant_projection_test_module.cpp'];files={n:base64.b64encode((repo/n).read_bytes()).decode() for n in names}
s='''from pathlib import Path
import json,os,subprocess,base64,datetime,hashlib
r=Path('/expt/builds/e40_prequant_native_v1');r.mkdir(exist_ok=False)
for name,data in FILES.items():
 p=r/'src'/name.removeprefix('xllm/');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(data))
(r/'source_manifest.json').write_text(json.dumps({n:hashlib.sha256(base64.b64decode(d)).hexdigest() for n,d in FILES.items()},indent=2))
worker="""from pathlib import Path
import os,json,subprocess,shlex,datetime,hashlib
r=Path('/expt/builds/e40_prequant_native_v1');b=Path('/workspace/xllm-mt-main/build/cmake.linux-x86_64-cpython-310');commands=json.loads((b/'compile_commands.json').read_text());base=shlex.split(next(x['command'] for x in commands if x['file'].endswith('/musa_aot_kernel.cpp')));idx=base.index('-o');base=base[:idx];base.insert(1,'-I'+str(r/'src'));base.insert(2,'-I/opt/venv/lib/python3.10/site-packages')
sources=['core/kernels/musa/deepseek_v41_dense_block32.cpp','core/kernels/musa/musa_aot_kernel.cpp','tests/core/kernels/musa/v41_prequant_projection_test_module.cpp'];objects=[];records=[];rc=1
try:
 for i,s in enumerate(sources):
  obj=r/(str(i)+'.o');cmd=base+['-o',str(obj),'-c',str(r/'src'/s)];records.append(cmd)
  with (r/(str(i)+'.compile.log')).open('xb') as f:p=subprocess.run(cmd,cwd=b,stdout=f,stderr=subprocess.STDOUT)
  assert p.returncode==0,'compile '+s;objects.append(str(obj))
 lib='/opt/venv/lib/python3.10/site-packages';vcp=Path('/workspace/xllm-mt-main/build/reuse_prebuilt_feihu1016/vcpkg_installed/x64-linux/lib');target=r/'v41_prequant_projection_test_module.so'
 cmd=['c++','-shared','-fPIC','-o',str(target)]+objects+['-L'+lib+'/torch/lib','-ltorch_python','-ltorch','-ltorch_cpu','-lc10','-L'+lib+'/torch_musa/lib','-lmusa_python','-L/usr/local/musa/lib','-lmusa','-lmudart',str(vcp/'libglog.a'),str(vcp/'libgflags.a'),'-lpthread','-ldl','-lz'];records.append(cmd)
 with (r/'link.log').open('xb') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 assert p.returncode==0,'link';p=subprocess.run(['/opt/venv/bin/python3','-c','import torch,torch_musa;import v41_prequant_projection_test_module;print("IMPORT_PASS")'],capture_output=True,text=True,env=dict(os.environ,PYTHONPATH=str(r),MUSA_VISIBLE_DEVICES='-1'));(r/'import.log').write_text(p.stdout+p.stderr);assert p.returncode==0,'import';rc=0
finally:
 (r/'commands.json').write_text(json.dumps(records,indent=2));v={'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}
 p=r/'v41_prequant_projection_test_module.so'
 if p.exists():v.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (r/'finished.json').write_text(json.dumps(v))
""";(r/'worker.py').write_text(worker)
py='/opt/venv/lib/python3.10/site-packages';env=dict(os.environ,LD_PRELOAD='',LD_LIBRARY_PATH=py+'/torch_musa/lib:'+py+'/torch/lib:'+py+'/tvm_ffi/lib:/usr/local/musa/lib',MUSA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2')
with (r/'worker.log').open('xb') as f:p=subprocess.Popen(['/opt/venv/bin/python3',str(r/'worker.py')],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'job.json').write_text(json.dumps(v));print(json.dumps(v))
'''.replace('FILES',repr(files));Path('/tmp/match_e40_start_native.py').write_text(s)
