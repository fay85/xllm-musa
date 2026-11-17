from pathlib import Path
import json,os,subprocess,datetime,time
r=Path('/expt/diagnostics/e50_native_gate_gemv_gpu_v1');r.mkdir(exist_ok=False);assert json.loads(Path('/expt/builds/e50_native_gate_gemv_v2/finished.json').read_text())['exit_code']==0
pre=json.loads(Path('/expt/00_control/sj10_preflight_p0.json').read_text());assert pre['gpu_free'] and time.time()-datetime.datetime.fromisoformat(pre['at']).timestamp()<180
worker='''from pathlib import Path
import json,os,subprocess,datetime
r=Path('/expt/diagnostics/e50_native_gate_gemv_gpu_v1');cfg=json.loads(Path('/expt/runs/p49_full_draft_confirm_on_v1/sj10/role_config.json').read_text());env=dict(os.environ);env.update(cfg['ranks'][0]['environment']);env.update(LD_PRELOAD='',PYTHONPATH='/expt/builds/e50_native_gate_gemv_v2:/expt/scripts:/expt/repos/xllm-musa:/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',MTHREADS_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',TVM_FFI_GPU_BACKEND='musa')
cmd=['python3','/expt/scripts/validate_v41_gate_gemv_v1.py','--fixture','/expt/inputs/e50_real_gate_weights.safetensors','--output',str(r/'result.json')]
with (r/'test.log').open('xb') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'command':cmd}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
''';(r/'worker.py').write_text(worker)
with (r/'worker.log').open('xb') as f:p=subprocess.Popen(['python3',str(r/'worker.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
v={'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'at':datetime.datetime.now().astimezone().isoformat()};(r/'job.json').write_text(json.dumps(v));print(json.dumps(v))
