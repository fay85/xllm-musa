from pathlib import Path
import os,subprocess,json,datetime
s=Path('/workspace/xllm-mt-main');b=s/'build/cmake.linux-x86_64-cpython-310';r=Path('/expt/builds/e53_native_rope_qdq_v1')
env=dict(os.environ);env.update(MUSAMAPPING_PATH='/usr/local/musa/tools/musamapping',MUSA_HOME='/usr/local/musa',CUDA_HOME='/usr/local/musa',
 LD_LIBRARY_PATH='/opt/venv/lib/python3.10/site-packages/torch_musa/lib:/opt/venv/lib/python3.10/site-packages/torch/lib:/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64:/usr/local/openmpi/lib',
 PATH=str(b/'musa_link_wrapper')+':/opt/venv/bin:/usr/local/musa/bin:/usr/bin:/bin',
 TVM_FFI_LIB_DIR='/opt/venv/lib/python3.10/site-packages/tvm_ffi/lib',
 LIBRARY_PATH=str(b/'xllm/core/layers/musa')+':/usr/local/musa/lib:/opt/intel/oneapi/mkl/lib/intel64')
targets=['xllm']


cmd=['bash',str(s/'scripts/ninja_safe.sh'),str(b),'-f','build.sg438.ninja','-j4']+targets
(r/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat(),'cmd':cmd,'env':{k:env[k] for k in ['PATH','MUSAMAPPING_PATH','LD_LIBRARY_PATH','LIBRARY_PATH']}},indent=2))
with (r/'build.log').open('wb') as f:p=subprocess.run(cmd,cwd=s,env=env,stdout=f,stderr=subprocess.STDOUT)
(r/'finished.json').write_text(json.dumps({'exit_code':p.returncode,'compiler_errors':sum(' error:' in line or 'FAILED:' in line for line in (r/'build.log').read_text(errors='replace').splitlines()),'at':datetime.datetime.now().astimezone().isoformat()}))
