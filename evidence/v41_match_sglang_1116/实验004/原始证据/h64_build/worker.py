from pathlib import Path
import json,subprocess,datetime,os
r=Path('/expt/builds/attention_replica_v1/sparse_h64');src=Path('/workspace/xllm-mt-main')
env=dict(os.environ,MUSA_VISIBLE_DEVICES='-1',MTHREADS_VISIBLE_DEVICES='void',PYTHONPATH='/donor',OMP_NUM_THREADS='1',MAX_JOBS='8')
cmd=['python3',str(src/'tools/musa_deepseek_v41/build_sparse_mate.py'),'--donor-root','/donor','--output',str(r/'artifact'),'--heads','64']
(r/'job.json').write_text(json.dumps({'pid':os.getpid(),'start_ticks':Path('/proc/self/stat').read_text().split()[21],'command':cmd,'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
with (r/'build.log').open('wb') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
(r/'finished.json').write_text(json.dumps({'exit_code':p.returncode,'at':datetime.datetime.now().astimezone().isoformat()}))
