from pathlib import Path
import json,subprocess,shlex,sysconfig,os,datetime
s=Path('/workspace/xllm-mt-main');b=s/'build/cmake.linux-x86_64-cpython-310';r=Path('/expt/builds/attention_replica_v1/small_checks');r.mkdir(exist_ok=True)
rows=json.loads((b/'compile_commands.json').read_text());row=next(x for x in rows if x['file'].endswith('/xllm.cpp'));cmd=shlex.split(row['command']);out=r/'test_module.o';source=r/'v41_attention_layout_test_module.cpp';cmd[cmd.index('-o')+1]=str(out);cmd[cmd.index('-c')+1]=str(source);cmd=[x for x in cmd if x!='-fPIE'];cmd+=['-fPIC','-O2'];commands=[cmd]
(r/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()},indent=2));rc=1
try:
 with (r/'compile.log').open('wb') as log:subprocess.run(cmd,cwd=b,stdout=log,stderr=subprocess.STDOUT,check=True)
 kernel=b/'xllm/core/kernels/musa/libcuda_kernels.a';torch=Path('/usr/local/lib/python3.10/dist-packages/torch/lib');musa=Path('/usr/local/lib/python3.10/dist-packages/torch_musa/lib');vcpkg=s/'build/reuse_prebuilt_feihu1016/vcpkg_installed/x64-linux/lib';target=r/('v41_attention_layout_test_module'+sysconfig.get_config_var('EXT_SUFFIX'))
 cmd=['/usr/bin/g++','-shared','-o',str(target),str(out),str(kernel),'-L/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib','-Wl,-rpath,/usr/local/lib/python3.10/dist-packages/tvm_ffi/lib','-ltvm_ffi','-lmccl','-L'+str(torch),'-L'+str(musa),'-L/usr/local/musa/lib','-Wl,-rpath,'+str(torch),'-Wl,-rpath,'+str(musa),'-ltorch','-ltorch_cpu','-ltorch_python','-lc10','-lmusa_kernels','-lmusa_python','-lmusa',str(vcpkg/'libglog.a'),str(vcpkg/'libgflags.a'),'-lpthread','-ldl'];commands.append(cmd)
 with (r/'link.log').open('wb') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 rc=0
finally:
 (r/'commands.json').write_text(json.dumps(commands,indent=2));(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
