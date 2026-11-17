from pathlib import Path
import json,os,subprocess,datetime,time
r=Path('/expt/diagnostics/e56_short_context_gpu_v1');b=Path('/expt/builds/e56_short_context_module_v1');cfg=json.loads(Path('/expt/runs/p55_draft_gate_off_v1/sj10/role_config.json').read_text());env=dict(os.environ);env.update(cfg['ranks'][0]['environment']);env.update(LD_PRELOAD='',PYTHONPATH=str(b)+':/expt/repos/xllm-musa:/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',MTHREADS_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',TVM_FFI_GPU_BACKEND='musa')
common=['--indexer-dir',env['XLLM_DSV41_INDEXER_AOT_DIR']];tasks=[('selection',['python3',str(b/'src/validate_short_context_selection.py'),'--output',str(r/'selection.json')]+common)]
for ratio in [1,2]:
 for cap in [512,1024]:
  label=f'attention_r{ratio}_c{cap}';tasks.append((label,['python3',str(b/'src/validate_short_context_attention.py'),'--output',str(r/(label+'.json')),'--sparse-so',env['XLLM_DSV41_MATE_SPARSE_SO'],'--ratio',str(ratio),'--capacity',str(cap)]+common))
results=[];rc=1
try:
 for label,cmd in tasks:
  with (r/(label+'.log')).open('xb') as f:
   p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);proc=Path('/proc',str(p.pid));start=proc.joinpath('stat').read_text().split()[21];exe=str(proc.joinpath('exe').resolve());(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':start,'exe':exe,'command':cmd}))
   try:code=p.wait(timeout=180)
   except subprocess.TimeoutExpired:
    assert proc.joinpath('stat').read_text().split()[21]==start and str(proc.joinpath('exe').resolve())==exe
    state=proc.joinpath('stat').read_text().split(') ')[1].split()[0]
    if state=='D':raise RuntimeError('D state; do not force cleanup')
    p.terminate()
    try:code=p.wait(timeout=30)
    except subprocess.TimeoutExpired:raise RuntimeError('Normal TERM did not release; inspect before any further action')
  results.append({'name':label,'exit_code':code});(r/'status.json').write_text(json.dumps(results));assert code==0,(label,code)
 rc=0
finally:
 (r/'finished.json').write_text(json.dumps({'exit_code':rc,'results':results,'at':datetime.datetime.now().astimezone().isoformat(),'accuracy_status':'仅通过基础功能检查，未完成模型精度验收'}))
