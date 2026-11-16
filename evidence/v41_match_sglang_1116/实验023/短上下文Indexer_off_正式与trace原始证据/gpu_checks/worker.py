from pathlib import Path
import json,os,subprocess,datetime
root=Path('/expt/builds/short_indexer_q4_checks_v1');r=root/'gpu_checks';cfg=json.loads(Path('/expt/runs/p21_replica_off_v2/sj9/role_config.json').read_text());seed=cfg['ranks'][0]['environment'];env=dict(os.environ,LD_LIBRARY_PATH=seed['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH=str(root)+':/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',TVM_FFI_GPU_BACKEND='musa')
args=['--indexer-dir',seed['XLLM_DSV41_INDEXER_AOT_DIR']];commands=[['python3',str(root/'validate_short_context_selection.py')]+args+['--output',str(r/'selection.json')]]
for ratio,capacity in [(1,512),(2,512),(1,1024),(2,1024)]:
 commands.append(['python3',str(root/'validate_short_context_attention.py')]+args+['--sparse-so',seed['XLLM_DSV41_MATE_SPARSE_SO'],'--ratio',str(ratio),'--capacity',str(capacity),'--output',str(r/f'attention_r{ratio}_c{capacity}.json')])
rc=1
try:
 for i,cmd in enumerate(commands):
  with (r/f'test_{i}.log').open('xb') as f:
   p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'command':cmd,'index':i}));code=p.wait()
  assert code==0,(i,code)
 rc=0
finally:(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
