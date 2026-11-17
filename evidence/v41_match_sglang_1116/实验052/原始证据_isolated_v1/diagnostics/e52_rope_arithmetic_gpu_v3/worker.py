from pathlib import Path
import json,os,subprocess,datetime,signal,time
r=Path('/expt/diagnostics/e52_rope_arithmetic_gpu_v3');cfg=json.loads(Path('/expt/runs/p51_bit_gather_reverse_off_v1/sj10/role_config.json').read_text());env=dict(os.environ);env.update(cfg['ranks'][0]['environment']);env.update(LD_PRELOAD='',PYTHONPATH='/expt/builds/e52_rope_table_module_v2:/expt/scripts:/expt/repos/xllm-musa:/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',MTHREADS_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',TVM_FFI_GPU_BACKEND='musa')
cmd=['python3','/expt/scripts/validate_v41_rope_arithmetic_v3.py','--output',str(r/'result.json')]
with (r/'test.log').open('xb') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);start=Path('/proc',str(p.pid),'stat').read_text().split()[21];(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':start,'command':cmd}));timeout=False
 try:rc=p.wait(timeout=240)
 except subprocess.TimeoutExpired:
  timeout=True;proc=Path('/proc',str(p.pid));assert proc.exists() and proc.joinpath('stat').read_text().split()[21]==start
  p.terminate()
  try:rc=p.wait(timeout=30)
  except subprocess.TimeoutExpired:rc=None
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'timed_out':timeout,'remaining_pid':p.pid if rc is None else None,'at':datetime.datetime.now().astimezone().isoformat()}))
