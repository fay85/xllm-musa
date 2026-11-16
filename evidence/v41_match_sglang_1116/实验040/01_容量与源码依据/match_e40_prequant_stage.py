from pathlib import Path
import base64
code=base64.b64encode(Path('/tmp/match_e40_prequant_probe.py').read_bytes()).decode()
s='''from pathlib import Path
import os,json,subprocess,datetime,base64,time
r=Path('/expt/diagnostics/e40_prequant_real_weights_v1');r.mkdir(exist_ok=False)
p=json.loads(Path('/expt/00_control/sj10_preflight_p0.json').read_text());assert p['gpu_free'] and time.time()-datetime.datetime.fromisoformat(p['at']).timestamp()<180
(r/'probe.py').write_bytes(base64.b64decode(CODE))
base=json.loads(Path('/expt/runs/p39_static_activation_on_v1/sj10/role_config.json').read_text())['ranks'][0]['environment'];env=dict(os.environ,PATH='/opt/venv/bin:/usr/local/musa/bin:'+os.environ.get('PATH',''),LD_PRELOAD='',LD_LIBRARY_PATH=base['LD_LIBRARY_PATH'],PYTHONPATH='/expt/builds/native_sg438_control/native_checks_v8:/expt/repos/xllm-musa:/workspace/sglang/python',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',TVM_FFI_GPU_BACKEND='musa',TRITON_CACHE_DIR=str(r/'triton_cache'))
worker="""from pathlib import Path
import subprocess,datetime,json,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0));r=Path('/expt/diagnostics/e40_prequant_real_weights_v1')
with (r/'probe.log').open('xb') as f:
 p=subprocess.Popen(['/opt/venv/bin/python3',str(r/'probe.py')],stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21]}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
""";(r/'worker.py').write_text(worker)
with (r/'worker.log').open('xb') as f:p=subprocess.Popen(['/opt/venv/bin/python3',str(r/'worker.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'job.json').write_text(json.dumps(v));print(json.dumps(v))
'''.replace('CODE',repr(code));Path('/tmp/match_e40_prequant_launch.py').write_text(s)
