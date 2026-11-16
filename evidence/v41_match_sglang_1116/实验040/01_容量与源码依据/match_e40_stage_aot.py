from pathlib import Path
import base64
repo=Path('王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/03_代码/xllm-musa')
code=base64.b64encode((repo/'tools/musa_deepseek_v41/build_dense_prequant_aot.py').read_bytes()).decode()
s='''from pathlib import Path
import json,os,base64,subprocess,datetime,hashlib
r=Path('/expt/builds/e40_prequant_aot_v1');r.mkdir(exist_ok=False);p=r/'build_dense_prequant_aot.py';p.write_bytes(base64.b64decode(CODE));out=Path('/expt/artifacts/e40_prequant_aot_v1');assert not out.exists()
shapes=[]
for m in [1,4,6]:
 for n,k,split in [(1280,5120,16),(512,5120,16),(32768,1280,4),(5120,8192,4),(4608,5120,4 if m==6 else 8),(5120,2304,8)]:shapes.append([m,n,k,split])
cmd=['/opt/venv/bin/python3',str(p),'--output',str(out)]
for shape in shapes:cmd+=['--shape']+[str(x) for x in shape]
py='/opt/venv/lib/python3.10/site-packages';env=dict(os.environ,PATH='/opt/venv/bin:/usr/local/musa/bin:'+os.environ.get('PATH',''),LD_PRELOAD='',PYTHONPATH='/expt/repos/xllm-musa',MUSA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='4',MAX_JOBS='4',TRITON_CACHE_DIR=str(r/'triton_cache'),LD_LIBRARY_PATH=':'.join(py+'/'+n+'/lib' for n in ['torch','torch_musa','tvm_ffi'])+':/usr/local/musa/lib')
(r/'manifest.json').write_text(json.dumps({'command':cmd,'shapes':shapes,'builder_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'base_builder_sha256':hashlib.sha256(Path('/expt/repos/xllm-musa/tools/musa_deepseek_v41/build_dense_block32_aot.py').read_bytes()).hexdigest(),'status':'CPU_OFFLINE_BUILD_ONLY_NOT_NATIVE_GPU_VALIDATED'},indent=2))
worker="""from pathlib import Path
import subprocess,json,datetime
r=Path('/expt/builds/e40_prequant_aot_v1');cmd=json.loads((r/'manifest.json').read_text())['command']
with (r/'build.log').open('xb') as f:
 p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
""";(r/'worker.py').write_text(worker)
with (r/'worker.log').open('xb') as f:p=subprocess.Popen(['/opt/venv/bin/python3',str(r/'worker.py')],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'job.json').write_text(json.dumps(v));print(json.dumps(v))
'''.replace('CODE',repr(code));Path('/tmp/match_e40_start_aot.py').write_text(s)
