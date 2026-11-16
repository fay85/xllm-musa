from pathlib import Path
import subprocess,json,os,datetime,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
r=Path('/expt/builds/prefix_checkpoint_v1/small_checks_v3');c=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj10/role_config.json').read_text());env=dict(os.environ,**c['ranks'][0]['environment']);env.update(MUSA_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_PRELOAD='',PYTHONPATH=str(r)+':/workspace/xllm-mt-main')
model=next(x.split('=',1)[1] for x in c['ranks'][0]['command'] if x.startswith('--model='))
cmd=['python3',str(r/'validate_prefix_engram.py'),'--output',str(r/'actual_engram.json'),'--metadata',str(Path(model)/'engram_hash_state.native'),'--library-directory',env['XLLM_DSV41_ENGRAM_HASH_AOT_DIR']]
with (r/'gpu_test.log').open('wb') as f:
 p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env);j={'worker_pid':os.getpid(),'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'command':cmd,'started_at':datetime.datetime.now().astimezone().isoformat()};(r/'gpu_job.json').write_text(json.dumps(j,indent=2));rc=p.wait()
(r/'gpu_finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
