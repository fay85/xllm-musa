from pathlib import Path
import json,os,subprocess,resource,datetime
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
r=Path('/expt/builds/draft_read_checks_v1/gpu_checks');c=json.loads(Path('/expt/runs/p7_spec2_full_verify_graph_v1/sj10/role_config.json').read_text());env=dict(os.environ,**c['ranks'][0]['environment']);env.update(MUSA_VISIBLE_DEVICES='0',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_PRELOAD='',PYTHONPATH=str(r.parent)+':/workspace/xllm-mt-main');model=next(x.split('=',1)[1] for x in c['ranks'][0]['command'] if x.startswith('--model='));rc=1
commands=[('draft_read',['python3',str(r/'validate_draft_read.py'),'--output',str(r/'draft_read.json'),'--sparse-library',env['XLLM_DSV41_MATE_SPARSE_SO']])]
for name,command in commands:
 with (r/(name+'.log')).open('xb') as f:
  p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT);v={'stage':name,'pid':p.pid,'start_ticks':Path('/proc').joinpath(str(p.pid),'stat').read_text().split()[21],'command':command,'at':datetime.datetime.now().astimezone().isoformat()};(r/'active.json').write_text(json.dumps(v,indent=2));(r/(name+'_job.json')).write_text(json.dumps(v,indent=2));rc=p.wait()
 (r/(name+'_finished.json')).write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
 if rc!=0:break
(r/'finished.json').write_text(json.dumps({'exit_code':rc,'at':datetime.datetime.now().astimezone().isoformat()},indent=2))
