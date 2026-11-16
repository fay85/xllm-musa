from pathlib import Path
import json,os,subprocess,datetime
r=Path('/expt/builds/replica_verify_q4_v1/sparse_pair_checks');cfg=json.loads(Path('/expt/runs/p18_projection_on_v1/sj9/role_config.json').read_text());env=dict(os.environ,LD_LIBRARY_PATH=cfg['ranks'][0]['environment']['LD_LIBRARY_PATH'],LD_PRELOAD='',PYTHONPATH='/expt/builds/attention_replica_v1/small_checks_v7:/workspace/xllm-mt-main',MUSA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
cmd=['python3','/workspace/xllm-mt-main/tools/musa_deepseek_v41/validate_sparse_library_pair.py','--h8',cfg['ranks'][0]['environment']['XLLM_DSV41_MATE_SPARSE_SO'],'--h64','/expt/artifacts/attention_replica_v1/lib/libdeepseek_v41_sparse_h64_bf16.so','--output',str(r/'result.json')]
with (r/'test.log').open('wb') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);(r/'child.json').write_text(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split()[21],'command':cmd}));rc=p.wait()
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()}))
