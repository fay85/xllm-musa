from pathlib import Path
import subprocess,json,os,datetime,resource
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
r=Path('/expt/builds/verify_graph_inputs_v1/other_ratios');cfg=json.loads(Path('/expt/runs/p2_full_decode_graph_v1/sj10/role_config.json').read_text());env=dict(os.environ,**cfg['ranks'][0]['environment']);env.update(MUSA_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_PRELOAD='',PYTHONPATH=str(r.parent)+':/workspace/xllm-mt-main');rc=1
for ratio in [0,1]:
 cmd=['python3',str(r/'validate_verify_graph_attention.py'),'--output',str(r/f'actual_attention_graph_ratio{ratio}.json'),'--sparse-so',env['XLLM_DSV41_MATE_SPARSE_SO'],'--indexer-dir',env['XLLM_DSV41_INDEXER_AOT_DIR'],'--ratio',str(ratio)]
 with (r/f'gpu_test_ratio{ratio}.log').open('xb') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);j={'worker_pid':os.getpid(),'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'ratio':ratio,'command':cmd,'at':datetime.datetime.now().astimezone().isoformat()};(r/'gpu_job.json').write_text(json.dumps(j,indent=2));rc=p.wait()
 (r/f'ratio{ratio}_finished.json').write_text(json.dumps({'exit_code':rc},indent=2))
 if rc!=0:break
(r/'gpu_finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
